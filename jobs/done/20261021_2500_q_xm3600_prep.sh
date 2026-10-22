# XM3600 다운로드·살펴보기 (모델 평가 없음, 약 10~20분)
set -eo pipefail
source ~/venvs/halio/bin/activate
python -m src.xm3600_prep download
python -m src.xm3600_prep inspect
du -sh /mnt/d/halio_data/data/xm3600
