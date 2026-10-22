# E23 레지스터 뉴런 진단 (10/17 22:30). 정확도·통계만, 학습 없음. COCO 제외 목록 96장 + CIFAR 검증 2000장. 약 15분
set -e
source ~/venvs/halio/bin/activate
python -m src.register_probe
