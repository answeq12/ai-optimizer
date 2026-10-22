# (재투입: 10/21 05:28 UTC 학습 300/444 스텝에서 멈춤 — PC 절전·WSL 정지로 추정. noisy 저장분은 재사용)
# E42 후보 ③: E39 방법(양자화 보정 학습)을 적응 모델(E33)에 적용 (약 1.5시간)
set -eo pipefail
source ~/venvs/halio/bin/activate
export HALIO_DA_FEAT=$(cat logs/da_feat_dir.txt)
python -m src.qat_tail noisy --base da --n 10000
timeout -k 60 3h python -m src.qat_tail train --base da --n 10000
export HALIO_MODELS_SUBDIR=models_qat_da; M=/mnt/d/halio_data/models_qat_da; mkdir -p $M/imp
python -m src.qat_tail export --base da
CAL=splits/calib_coco_64_s0.txt; F=$M/imp/split_SPL-b567-k32.fp32.onnx
[ -f $F ] || python -m src.split_outliers --blocks 5,6,7 --k 32 --calib-list $CAL --out $F
[ -f $M/imp/SPL-b567-k32.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --src $F --exclude-nodes-file $F.hot_nodes.txt --calib-list $CAL --out $M/imp/SPL-b567-k32.onnx
for c in "QAT-DA-FP32 $M/clip_vitb16_image_fp32.onnx fp32" "QAT-DA-SPL $M/imp/SPL-b567-k32.onnx static"; do set -- $c
  python -m src.evaluate --model $2 --code $1 --quant-mode $3 --eval-set val | tail -1
  python -m src.coco_retrieval eval --model $2 --code $1 --split select | tail -1
done
unset HALIO_MODELS_SUBDIR
python -m src.coco_retrieval table --ref A0 --codes A0 SPL-b567-k32 DA-A0 DA-SPL QAT-DA-FP32 QAT-DA-SPL --split select --margin 2.0 --tag e42_qatda
