# (재투입: 블록7 텐서 확인 기준을 상대 오차로 수정)
# E39 양자화 보정 학습 (원본 가중치, 약 1.5~2시간): INT8 블록7 출력 저장 → fake-quant 꼬리 증류 → 변환·SPL → select·CIFAR val
set -eo pipefail
source ~/venvs/halio/bin/activate
export HALIO_DA_FEAT=$(cat logs/da_feat_dir.txt)
python -m src.qat_tail noisy --base orig --n 10000
timeout -k 60 3h python -m src.qat_tail train --base orig --n 10000
export HALIO_MODELS_SUBDIR=models_qat_orig; M=/mnt/d/halio_data/models_qat_orig; mkdir -p $M/imp
python -m src.qat_tail export --base orig
CAL=splits/calib_coco_64_s0.txt; F=$M/imp/split_SPL-b567-k32.fp32.onnx
[ -f $F ] || python -m src.split_outliers --blocks 5,6,7 --k 32 --calib-list $CAL --out $F
[ -f $M/imp/SPL-b567-k32.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --src $F --exclude-nodes-file $F.hot_nodes.txt --calib-list $CAL --out $M/imp/SPL-b567-k32.onnx
cmp -s /mnt/d/halio_data/models/imp/split_SPL-b567-k32.fp32.onnx.hot_nodes.txt $F.hot_nodes.txt && echo "보호 노드 원본과 동일" || echo "보호 노드 다름"
python -m src.evaluate --model $M/imp/SPL-b567-k32.onnx --code QAT-SPL --quant-mode static --eval-set val | tail -1
python -m src.coco_retrieval eval --model $M/imp/SPL-b567-k32.onnx --code QAT-SPL --split select | tail -1
python -m src.evaluate --model $M/clip_vitb16_image_fp32.onnx --code QAT-FP32 --quant-mode fp32 --eval-set val | tail -1
python -m src.coco_retrieval eval --model $M/clip_vitb16_image_fp32.onnx --code QAT-FP32 --split select | tail -1
unset HALIO_MODELS_SUBDIR
python -m src.coco_retrieval table --ref A0 --codes A0 SPL-b567-k32 QAT-FP32 QAT-SPL --split select --margin 2.0 --tag e39
python - <<'PY'
import json
from src.da_gate import diff_ci
m,lo,hi=diff_ci("QAT-SPL","SPL-b567-k32","select")
acc=lambda c: json.load(open(f"results/preds/{c}-val.json"))["top1_acc"]*100
dc=acc("QAT-SPL")-acc("SPL-b567-k32"); ok=lo>0 and dc>=-0.5
print(f"[E39] select QAT-SPL − SPL R@1 {m:+.2f} [{lo:+.2f}, {hi:+.2f}], CIFAR val {dc:+.2f}%p → {'채택 규칙 충족' if ok else '미충족'}")
json.dump({"diff":m,"ci":[lo,hi],"cifar_val_diff":dc,"pass":ok},open("results/e39_decision.json","w"),indent=2)
PY
