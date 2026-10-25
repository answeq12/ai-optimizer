# E45 양자화 보정 학습(E39 방식)을 DA3 에 적용 (약 2시간): INT8 블록7 출력 저장 → 보정 학습 → models_qat_da3 변환·SPL → select·CIFAR val, DA3-SPL 과 비교
set -eo pipefail
source ~/venvs/halio/bin/activate
export HALIO_DA_FEAT=$(cat logs/da_feat_dir.txt)
python -m src.qat_tail noisy --base da3 --n 10000
timeout -k 60 3h python -m src.qat_tail train --base da3 --n 10000
export HALIO_MODELS_SUBDIR=models_qat_da3; M=/mnt/d/halio_data/models_qat_da3; mkdir -p $M/imp
python -m src.qat_tail export --base da3
CAL=splits/calib_coco_64_s0.txt; F=$M/imp/split_SPL-b567-k32.fp32.onnx
[ -f $F ] || python -m src.split_outliers --blocks 5,6,7 --k 32 --calib-list $CAL --out $F
[ -f $M/imp/SPL-b567-k32.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --src $F --exclude-nodes-file $F.hot_nodes.txt --calib-list $CAL --out $M/imp/SPL-b567-k32.onnx
cmp -s /mnt/d/halio_data/models/imp/split_SPL-b567-k32.fp32.onnx.hot_nodes.txt $F.hot_nodes.txt && echo "보호 노드 원본과 동일" || echo "보호 노드 다름"
for c in "QAT-DA3-FP32 $M/clip_vitb16_image_fp32.onnx fp32" "QAT-DA3-SPL $M/imp/SPL-b567-k32.onnx static"; do set -- $c
  python -m src.evaluate --model $2 --code $1 --quant-mode $3 --eval-set val | tail -1
  python -m src.coco_retrieval eval --model $2 --code $1 --split select | tail -1
done
python -m src.coco_retrieval table --ref DA3-SPL --codes DA3-SPL DA3-A0 QAT-DA3-FP32 QAT-DA3-SPL --split select --margin 2.0 --tag e45_vs_da3
