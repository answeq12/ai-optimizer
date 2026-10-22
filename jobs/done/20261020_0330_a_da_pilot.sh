# E33 P0 파일럿 (약 30분~1시간): 앞/뒤 나눔 검증, train2017 2,000장 받기, 블록7 저장 속도, 캡션 임베딩, 학습 30스텝 속도·메모리
set -e
source ~/venvs/halio/bin/activate
echo "== 디스크/메모리"; df -h ~ /mnt/d | cat; free -g | cat
AV=$(df -BG --output=avail ~ | tail -1 | tr -dc 0-9)
if [ "$AV" -gt 40 ]; then export HALIO_DA_FEAT=$HOME/halio_da_feat; else export HALIO_DA_FEAT=/mnt/d/halio_data/da_feat; fi
echo "특징 저장 위치: $HALIO_DA_FEAT (홈 여유 ${AV}G)"; echo "$HALIO_DA_FEAT" > logs/da_feat_dir.txt
python -m src.da_cache check
t=$(date +%s); python -m src.da_data --n 2000; echo "다운로드 2000장: $(( $(date +%s)-t ))s"
python -m src.da_cache images --set train --n 2000
python -m src.da_cache text --n 2000
timeout -k 60 40m python -m src.da_train --n 2000 --max-steps 30
echo "== 파일럿 끝"; cat logs/da_train_pilot.json
