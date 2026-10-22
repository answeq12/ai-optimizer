# E33 본 학습 (약 3시간): train2017 30,000장 받기 → 블록7 저장(train·select·CIFAR val) → 캡션 임베딩 → 2에폭 학습 → α 탐색·판단 ①
set -e
source ~/venvs/halio/bin/activate
CF=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9); echo "C 드라이브 여유 ${CF}G"
if [ "$CF" -lt 40 ]; then echo /mnt/d/halio_data/da_feat > logs/da_feat_dir.txt; fi
export HALIO_DA_FEAT=$(cat logs/da_feat_dir.txt); echo "특징 저장 위치: $HALIO_DA_FEAT"
t=$(date +%s); python -m src.da_data --workers 32; echo "다운로드: $(( $(date +%s)-t ))s"
python -m src.da_cache images --set train --n 30000
python -m src.da_cache images --set select
python -m src.da_cache images --set cifarval
python -m src.da_cache text --n 30000
timeout -k 60 4h python -m src.da_train --n 30000
python -m src.da_sweep --ckpt da_tail_ep2.pt
