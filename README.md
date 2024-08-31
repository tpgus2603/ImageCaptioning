# ImageCaptioning

분산 머신러닝으로 멀티 모달 데이터(이미지 ,텍스트) 처리 및 이미지 캡셔닝 모델 구현 , 멀티 GPU를 활용하기 위해

torch.distributed를 이용하여분산 시스템을 구축한 코드가 pair로 존재합니다.

주피터 노트북 환경에서 동작하는 코드이므로 터미널을 통해 파이썬 코드를 실행시 적절한 수정이 필요합니다.

멀티 gpu에 관련된 코드를 실행하기 위해선 터미널 환경에서 
python -m torch.distributed.launch --nproc_per_node=4 test.py  명령어를 통해 실행하세요
