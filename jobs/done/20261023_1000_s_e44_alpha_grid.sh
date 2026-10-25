# E44 α 따로(이미지·텍스트) + ep1·ep2 평균 격자 (재학습 없음, 약 1~1.5시간) → 선택 구성 models_da4 변환·SPL·select/CIFAR val, DA3-SPL 과 비교
set -eo pipefail
source ~/venvs/halio/bin/activate
export HALIO_DA_FEAT=$(cat logs/da_feat_dir.txt)
python -m src.da_grid --tag e37 --out e44
export HALIO_MODELS_SUBDIR=models_da4; M=/mnt/d/halio_data/models_da4; mkdir -p $M/imp
python -m src.da_text export --tag e44
CAL=splits/calib_coco_64_s0.txt; F=$M/imp/split_SPL-b567-k32.fp32.onnx
[ -f $F ] || python -m src.split_outliers --blocks 5,6,7 --k 32 --calib-list $CAL --out $F
[ -f $M/imp/SPL-b567-k32.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --src $F --exclude-nodes-file $F.hot_nodes.txt --calib-list $CAL --out $M/imp/SPL-b567-k32.onnx
cmp -s /mnt/d/halio_data/models/imp/split_SPL-b567-k32.fp32.onnx.hot_nodes.txt $F.hot_nodes.txt && echo "보호 노드 원본과 동일" || echo "보호 노드 다름"
python -m src.evaluate --model $M/clip_vitb16_image_fp32.onnx --code DA4-A0 --quant-mode fp32 --eval-set val | tail -1
python -m src.evaluate --model $M/imp/SPL-b567-k32.onnx --code DA4-SPL --quant-mode static --eval-set val | tail -1
python -m src.coco_retrieval eval --model $M/clip_vitb16_image_fp32.onnx --code DA4-A0 --split select | tail -1
python -m src.coco_retrieval eval --model $M/imp/SPL-b567-k32.onnx --code DA4-SPL --split select | tail -1
python -m src.coco_retrieval table --ref DA3-SPL --codes DA3-SPL DA4-A0 DA4-SPL --split select --margin 2.0 --tag e44_vs_da3
