# fc2 민감도 원인 가설 검증: 가중치 행렬곱 49개의 입력 활성값 분포 (FP32, 64장). 약 5분
set -e
source ~/venvs/halio/bin/activate
python -m src.act_stats --calib-list splits/calib_coco_64_s0.txt
python -m src.act_stats --calib-list splits/calib_cifar_64_s0.txt --tag cifar > /dev/null
