# E36 텍스트 인코더도 학습 (약 2.5~3시간): 나눔 검증 → 텍스트 앞부분 저장 → 2에폭 학습 → α 탐색 → (선택 시) models_da2 변환·SPL·select/CIFAR val 평가
set -eo pipefail
source ~/venvs/halio/bin/activate
export HALIO_DA_FEAT=$(cat logs/da_feat_dir.txt)
python -m src.da_text check
python -m src.da_text cache --n 30000
timeout -k 60 4h python -m src.da_text train --n 30000 --tag e36
python -m src.da_text sweep --tag e36 --ckpt da2_e36_ep2.pt
[ -f /mnt/d/halio_data/models_da/da2_chosen_e36.pt ] || { echo "선택 없음 → 종료"; exit 0; }
export HALIO_MODELS_SUBDIR=models_da2; M=/mnt/d/halio_data/models_da2; mkdir -p $M/imp
python -m src.da_text export --tag e36
CAL=splits/calib_coco_64_s0.txt; F=$M/imp/split_SPL-b567-k32.fp32.onnx
[ -f $F ] || python -m src.split_outliers --blocks 5,6,7 --k 32 --calib-list $CAL --out $F
[ -f $M/imp/SPL-b567-k32.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --src $F --exclude-nodes-file $F.hot_nodes.txt --calib-list $CAL --out $M/imp/SPL-b567-k32.onnx
cmp -s /mnt/d/halio_data/models/imp/split_SPL-b567-k32.fp32.onnx.hot_nodes.txt $F.hot_nodes.txt && echo "보호 노드 원본과 동일" || echo "보호 노드 다름"
python -m src.evaluate --model $M/clip_vitb16_image_fp32.onnx --code DA2-A0 --quant-mode fp32 --eval-set val | tail -1
python -m src.evaluate --model $M/imp/SPL-b567-k32.onnx --code DA2-SPL --quant-mode static --eval-set val | tail -1
python -m src.coco_retrieval eval --model $M/clip_vitb16_image_fp32.onnx --code DA2-A0 --split select | tail -1
python -m src.coco_retrieval eval --model $M/imp/SPL-b567-k32.onnx --code DA2-SPL --split select | tail -1
python -m src.coco_retrieval table --ref A0 --codes A0 DA-A0 DA-SPL DA2-A0 DA2-SPL --split select --margin 2.0 --tag da2
