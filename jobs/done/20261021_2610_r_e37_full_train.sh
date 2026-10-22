# E37 train2017 전체(118,287장)로 DA2 재학습 (약 8시간): 사진 받기 → 이미지·텍스트 앞부분 저장(앞 30k 재사용) → 2에폭 → α 탐색 → models_da3 변환·SPL → select·CIFAR val 비교
set -eo pipefail
source ~/venvs/halio/bin/activate
export HALIO_DA_FEAT=$(cat logs/da_feat_dir.txt)
N=118287
python -m src.da_data --n $N
python -m src.da_cache images --set train --n $N
python -m src.da_text cache --n $N
timeout -k 60 7h python -m src.da_text train --n $N --tag e37
python -m src.da_text sweep --tag e37 --ckpt da2_e37_ep2.pt
[ -f /mnt/d/halio_data/models_da/da2_chosen_e37.pt ] || { echo "선택 없음 → 종료"; exit 0; }
export HALIO_MODELS_SUBDIR=models_da3; M=/mnt/d/halio_data/models_da3; mkdir -p $M/imp
python -m src.da_text export --tag e37
CAL=splits/calib_coco_64_s0.txt; F=$M/imp/split_SPL-b567-k32.fp32.onnx
[ -f $F ] || python -m src.split_outliers --blocks 5,6,7 --k 32 --calib-list $CAL --out $F
[ -f $M/imp/SPL-b567-k32.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --src $F --exclude-nodes-file $F.hot_nodes.txt --calib-list $CAL --out $M/imp/SPL-b567-k32.onnx
cmp -s /mnt/d/halio_data/models/imp/split_SPL-b567-k32.fp32.onnx.hot_nodes.txt $F.hot_nodes.txt && echo "보호 노드 원본과 동일" || echo "보호 노드 다름"
python -m src.evaluate --model $M/clip_vitb16_image_fp32.onnx --code DA3-A0 --quant-mode fp32 --eval-set val | tail -1
python -m src.evaluate --model $M/imp/SPL-b567-k32.onnx --code DA3-SPL --quant-mode static --eval-set val | tail -1
python -m src.coco_retrieval eval --model $M/clip_vitb16_image_fp32.onnx --code DA3-A0 --split select | tail -1
python -m src.coco_retrieval eval --model $M/imp/SPL-b567-k32.onnx --code DA3-SPL --split select | tail -1
python -m src.coco_retrieval table --ref A0 --codes A0 DA2-A0 DA2-SPL DA3-A0 DA3-SPL --split select --margin 2.0 --tag e37
python -m src.coco_retrieval table --ref DA2-SPL --codes DA2-SPL DA3-A0 DA3-SPL --split select --margin 2.0 --tag e37_vs_da2
