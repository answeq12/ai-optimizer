# E46 CC3M 10만장 추가 학습 (약 12시간): 안 쓰는 캐시 정리 → CC3M 받기 → 블록7 저장 → 기준(A0·DA3-SPL) CC3M val → COCO+CC3M 2에폭 → α(COCO·CC3M 평균) → models_da5 → 비교
set -eo pipefail
source ~/venvs/halio/bin/activate
export HALIO_DA_FEAT=$(cat logs/da_feat_dir.txt); FD=$HALIO_DA_FEAT
# 1) 안 쓰는 중간 캐시 정리(모두 다시 만들 수 있음): train_30000(=train_118287 앞부분), ttrain_30000(E36 텍스트 저장), noisy7_*(QAT 끝), train_2000(파일럿)
rm -f $FD/train_30000.f16 $FD/ttrain_30000.f16 $FD/ttrain_30000_eot.npy $FD/noisy7_*.f16 $FD/train_2000.f16
df -h $HOME /mnt/c | cat
CFREE=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9); echo "C 여유 ${CFREE}GB"; [ "$CFREE" -ge 15 ] || { echo "C 여유 부족 → 중단"; exit 1; }
# 2) 데이터
python -m src.cc3m_eval prepare --train-shards 22 --val-shards 3
python -m src.da_cache images --set cc3mval
python -m src.da_cache images --set cc3m
CFREE=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9); echo "CC3M 저장 후 C 여유 ${CFREE}GB"
# 3) 기준 CC3M val (학습 전)
python -m src.cc3m_eval text
python -m src.cc3m_eval text --tag da3 --text-model /mnt/d/halio_data/models_da3/text_state.pt
python -m src.cc3m_eval eval A0=/mnt/d/halio_data/models/clip_vitb16_image_fp32.onnx DA3-A0=/mnt/d/halio_data/models_da3/clip_vitb16_image_fp32.onnx:da3 DA3-SPL=/mnt/d/halio_data/models_da3/imp/SPL-b567-k32.onnx:da3
# 4) 학습·α
timeout -k 60 11h python -m src.da_text train-mix --n 118287 --tag e46
python -m src.da_text sweep --tag e46 --ckpt da2_e46_ep2.pt --cc3m
[ -f /mnt/d/halio_data/models_da/da2_chosen_e46.pt ] || { echo "선택 없음 → 종료"; exit 0; }
# 5) 변환·평가
export HALIO_MODELS_SUBDIR=models_da5; M=/mnt/d/halio_data/models_da5; mkdir -p $M/imp
python -m src.da_text export --tag e46
CAL=splits/calib_coco_64_s0.txt; F=$M/imp/split_SPL-b567-k32.fp32.onnx
[ -f $F ] || python -m src.split_outliers --blocks 5,6,7 --k 32 --calib-list $CAL --out $F
[ -f $M/imp/SPL-b567-k32.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --src $F --exclude-nodes-file $F.hot_nodes.txt --calib-list $CAL --out $M/imp/SPL-b567-k32.onnx
cmp -s /mnt/d/halio_data/models/imp/split_SPL-b567-k32.fp32.onnx.hot_nodes.txt $F.hot_nodes.txt && echo "보호 노드 원본과 동일" || echo "보호 노드 다름"
for c in "DA5-A0 $M/clip_vitb16_image_fp32.onnx fp32" "DA5-SPL $M/imp/SPL-b567-k32.onnx static"; do set -- $c
  python -m src.evaluate --model $2 --code $1 --quant-mode $3 --eval-set val | tail -1
  python -m src.coco_retrieval eval --model $2 --code $1 --split select | tail -1
done
python -m src.coco_retrieval table --ref DA3-SPL --codes DA3-SPL DA5-A0 DA5-SPL --split select --margin 2.0 --tag e46_vs_da3
unset HALIO_MODELS_SUBDIR
python -m src.cc3m_eval text --tag da5 --text-model $M/text_state.pt
python -m src.cc3m_eval eval DA5-A0=$M/clip_vitb16_image_fp32.onnx:da5 DA5-SPL=$M/imp/SPL-b567-k32.onnx:da5
python -m src.cc3m_eval table --ref DA3-SPL --codes A0 DA3-A0 DA3-SPL DA5-A0 DA5-SPL
