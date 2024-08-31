import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import BlipForConditionalGeneration, BlipProcessor, get_scheduler
from PIL import Image
import pandas as pd
from datasets import Dataset
import evaluate
from nltk.translate.bleu_score import corpus_bleu
import os
import time
from torch.nn.utils.rnn import pad_sequence

# Initialize the process group for DDP
def setup():
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl')

# Clean up DDP
def cleanup():
    dist.destroy_process_group()

# Set device to GPU if available
def main(local_rank):
    setup()
    device = torch.device(f'cuda:{local_rank}')
    torch.manual_seed(0)

    # Load the BLIP model and processor
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

    # Wrap the model in DDP with find_unused_parameters=True
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Load the dataset
    CAPTIONS_FILE = "/home/jupyter/kaggle/input/flickr8k/captions.txt"
    IMAGES_DIR = '/home/jupyter/kaggle/input/flickr8k/Images'

    df = pd.read_csv(CAPTIONS_FILE)
    df['image'] = df['image'].apply(lambda x: f'{IMAGES_DIR}/{x}')

    # Split the dataset
    total_length = len(df)
    train_size = int(total_length * 0.7)
    val_size = int(total_length * 0.2)
    test_size = total_length - train_size - val_size

    train_df = df.iloc[:train_size, :]
    val_df = df.iloc[train_size:train_size + val_size, :]
    test_df = df.iloc[train_size + val_size:, :]

    # Convert to HuggingFace Dataset
    processed_train = Dataset.from_pandas(train_df)
    processed_val = Dataset.from_pandas(val_df)
    processed_test = Dataset.from_pandas(test_df)

    # Dataset class definition
    class Flickr8kDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, processor):
            self.dataset = dataset
            self.processor = processor

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image_path = self.dataset[idx]['image']
            caption = self.dataset[idx]['caption']

            # Load the image and preprocess it using the processor
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, text=caption, return_tensors="pt", padding="max_length", truncation=True)

            # Convert to tensors without moving to device (handled later)
            pixel_values = inputs.pixel_values.squeeze(0)
            input_ids = inputs.input_ids.flatten()
            attention_mask = inputs.attention_mask.flatten()

            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "caption": caption  # Include the caption
            }

    # Custom collate function to handle varying sequence lengths
    def custom_collate_fn(batch):
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
        captions = [item['caption'] for item in batch]  # Collect captions

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "caption": captions  # Include captions in the batch
        }

    # Create data loaders with DistributedSampler
    train_sampler = DistributedSampler(processed_train, num_replicas=dist.get_world_size(), rank=local_rank)
    val_sampler = DistributedSampler(processed_val, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=False)

    train_loader = DataLoader(Flickr8kDataset(processed_train, processor), batch_size=4, sampler=train_sampler, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(Flickr8kDataset(processed_val, processor), batch_size=4, sampler=val_sampler, num_workers=0, collate_fn=custom_collate_fn)

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 10
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_epochs * len(train_loader)
    )

    # Early stopping class
    class EarlyStopping:
        def __init__(self, patience=3, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    # Training function with epoch timing
    def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=5, patience=2):
        early_stopping = EarlyStopping(patience=patience)
        model.train()

        for epoch in range(num_epochs):
            start_time = time.time()  # Start the timer at the beginning of the epoch

            total_loss = 0
            for batch in train_loader:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                optimizer.zero_grad()
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # Calculate the time taken for this epoch
            epoch_time = time.time() - start_time

            if local_rank == 0:  # Print loss on rank 0 only
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} seconds")

            # Validation after each epoch
            val_loss = evaluate_model(model, val_loader)

            # Early stopping check
            early_stopping(val_loss)
            if early_stopping.early_stop:
                if local_rank == 0:
                    print("Early stopping triggered")
                break

        if local_rank == 0:
            # Save the trained model
            model.module.save_pretrained("./blip-finetuned")

    # Evaluation function
    def evaluate_model(model, val_loader):
        model.eval()
        total_loss = 0
        for batch in val_loader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        if local_rank == 0:
            print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    # Train the model with early stopping and epoch timing
    train_model(model, train_loader, val_loader, optimizer, lr_scheduler, num_epochs=5, patience=2)

    cleanup()

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    main(local_rank)
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


# Set device to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = BlipForConditionalGeneration.from_pretrained("./blip-finetuned").to(device)
#processor = BlipProcessor.from_pretrained("./blip-finetuned")

import torch
from transformers import BlipForConditionalGeneration, BlipProcessor


# Set device to GPU if available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = BlipForConditionalGeneration.from_pretrained("./blip-finetuned").to(device)
#processor = BlipProcessor.from_pretrained("./blip-finetuned")

# Function to test the model and generate captions
def test_model(model, test_loader):
    model.eval()
    predictions = []
    references = []

    for batch in test_loader:
        pixel_values = batch['pixel_values'].to(device)
        captions = batch['caption']  # Captions are now correctly accessed from the batch

        with torch.no_grad():
            output_ids = model.generate(pixel_values, max_length=20, num_beams=3, early_stopping=True)
            preds = processor.batch_decode(output_ids, skip_special_tokens=True)
            predictions.extend(preds)
            references.extend(captions)

    return predictions, references


# Test the model
predictions, references = test_model(model, test_loader)

# Function to generate a caption for an image
def generate_caption(image_path):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate the caption
    with torch.no_grad():
        output_ids = model.generate(inputs['pixel_values'], max_length=20, num_beams=3, early_stopping=True)
    
    # Decode the generated caption
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return caption

# BLEU score calculation function
def compute_bleu(predictions, references):
    # Compute BLEU scores using different weights
    bleu_score_1 = corpus_bleu(references, predictions, weights=(0.3, 0.3, 0.3, 0))
    bleu_score_2 = corpus_bleu(references, predictions, weights=(0.25, 0.25, 0.25, 0.25))

    # Return BLEU scores as a dictionary
    return {
        'bleu_1': bleu_score_1,
        'bleu_2': bleu_score_2
    }

# ROUGE score calculation function
def compute_rouge(predictions, references):
    rouge_metric = evaluate.load("rouge")
    results = rouge_metric.compute(predictions=predictions, references=references)
    rouge_1 = results['rouge1']
    rouge_2 = results['rouge2']
    rouge_l = results['rougeL']
    return rouge_1, rouge_2, rouge_l

# Calculate BLEU and ROUGE scores
def get_scores(correct_caps, gen_caption):
    correct_caps_list = [correct_caps.split("\n")]
    predictions = [gen_caption]

    # Compute BLEU scores
    bleu_scores = compute_bleu(predictions, correct_caps_list)

    # Compute ROUGE scores
    rouge_1, rouge_2, rouge_l = compute_rouge(predictions, correct_caps_list)

    return bleu_scores, rouge_1, rouge_2, rouge_l

# Function to display the image, generated caption, and scores
def image_display(filepath, correct_caps, gen_caption, bleu_scores=None, rouge_1=None, rouge_2=None, rouge_l=None, font_size=7, dpi=300, save_img=False):
    img_color = cv2.imread(filepath, 1)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))

    correct_caps_string = f"> Correct Captions:\n{correct_caps}"
    generated_caption_string = f"> Generated Caption:\n{gen_caption}"

    scores_string = "> Scores:\n"
    if bleu_scores is not None:
        scores_string += f"BLEU-1: {bleu_scores['bleu_1']:.4f} | "
        scores_string += f"BLEU-2: {bleu_scores['bleu_2']:.4f} | "
    if rouge_1 is not None and rouge_2 is not None and rouge_l is not None:
        scores_string += f"ROUGE-1: {rouge_1:.4f} | ROUGE-2: {rouge_2:.4f} | ROUGE-L: {rouge_l:.4f}"

    title_string = f"{correct_caps_string}\n\n{generated_caption_string}\n\n{scores_string.strip('| ')}"
    plt.title(title_string, fontsize=font_size, wrap=True, loc='left')

    if save_img:
        plt.savefig(f"{os.path.splitext(os.path.basename(filepath))[0]}_generation", bbox_inches='tight', dpi=dpi)
    plt.show()

# Generate caption for an image and display it with BLEU and ROUGE scores
def get_caption_for_image(filepath, correct_caps, font_size=7, dpi=300, save_img=False):
    gen_caption = generate_caption(filepath)
    bleu_scores, rouge_1, rouge_2, rouge_l = get_scores(correct_caps, gen_caption)
    image_display(filepath, correct_caps, gen_caption, bleu_scores, rouge_1, rouge_2, rouge_l, font_size, dpi, save_img)

# Group test data and display example captions
grouped_test_df = test_df.groupby('image')['caption'].agg(lambda x: '\n'.join(x)).reset_index()
grouped_test_ds = Dataset.from_pandas(grouped_test_df)

examples = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
for i in examples:
    instance = grouped_test_ds.__getitem__(i)
    image_path = instance['image']
    caption = instance['caption']
    print(f"\n\nImage {i} --> {os.path.splitext(os.path.basename(image_path))[0]}")
    get_caption_for_image(image_path, caption, font_size=7, dpi=500, save_img=True)

import numpy as np

# Lists to store the BLEU and ROUGE scores
bleu_scores_list = []
rouge_1_list = []
rouge_2_list = []
rouge_l_list = []

# Loop through the test dataset
for i in range(len(grouped_test_ds)):
    instance = grouped_test_ds.__getitem__(i)
    image_path = instance['image']
    caption = instance['caption']

    # Generate caption for the image
    gen_caption = generate_caption(image_path)

    # Get BLEU and ROUGE scores
    bleu_scores, rouge_1, rouge_2, rouge_l = get_scores(caption, gen_caption)

    # Save the scores to the lists
    bleu_scores_list.append(bleu_scores)
    rouge_1_list.append(rouge_1)
    rouge_2_list.append(rouge_2)
    rouge_l_list.append(rouge_l)

# Calculate the average of each score
avg_bleu_1 = np.mean([score['bleu_1'] for score in bleu_scores_list])
avg_bleu_2 = np.mean([score['bleu_2'] for score in bleu_scores_list])
avg_rouge_1 = np.mean(rouge_1_list)
avg_rouge_2 = np.mean(rouge_2_list)
avg_rouge_l = np.mean(rouge_l_list)

# Print the average scores
print(f"\nAverage BLEU-1: {avg_bleu_1:.4f}")
print(f"Average BLEU-2: {avg_bleu_2:.4f}")
print(f"Average ROUGE-1: {avg_rouge_1:.4f}")
print(f"Average ROUGE-2: {avg_rouge_2:.4f}")
print(f"Average ROUGE-L: {avg_rouge_l:.4f}")