# E33 변환·양자화·평가 (약 1시간): 판단 ① 통과 시에만. FP32 ONNX + P1 → 같은 SPL 처방 → select 평가·판단 ② → 통과 시 confirm·CIFAR test 한 번 → 교체 규칙
set -eo pipefail
source ~/venvs/halio/bin/activate
[ -f /mnt/d/halio_data/models_da/da_tail_chosen.pt ] || { echo "판단 ① 미통과(또는 학습 실패) → 이 작업은 건너뜀"; exit 0; }
export HALIO_MODELS_SUBDIR=models_da
M=$(python -m src.common models-dir); mkdir -p $M/imp; echo "모델 폴더 $M"
CAL=splits/calib_coco_64_s0.txt
python -m src.da_export
F=$M/imp/split_SPL-b567-k32.fp32.onnx
[ -f "$F" ] || python -m src.split_outliers --blocks 5,6,7 --k 32 --calib-list $CAL --out "$F"
[ -f $M/imp/SPL-b567-k32.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel \
    --src "$F" --exclude-nodes-file "$F.hot_nodes.txt" --calib-list $CAL --out $M/imp/SPL-b567-k32.onnx
O=/mnt/d/halio_data/models/imp/split_SPL-b567-k32.fp32.onnx.hot_nodes.txt
if [ -f "$O" ]; then cmp -s "$O" "$F.hot_nodes.txt" && echo "보호 노드 목록: 원본과 동일" || echo "보호 노드 목록: 원본과 다름(확인 필요)"; fi
python -m src.act_stats --tag da > /dev/null; grep -E "fc2" results/act_stats_da.md | tail -4 || true
python -m src.evaluate --model $M/clip_vitb16_image_fp32.onnx --code DA-A0 --quant-mode fp32 --eval-set val | tail -1
python -m src.evaluate --model $M/imp/SPL-b567-k32.onnx --code DA-SPL --quant-mode static --eval-set val | tail -1
python -m src.coco_retrieval eval --model $M/clip_vitb16_image_fp32.onnx --code DA-A0 --split select | tail -1
python -m src.coco_retrieval eval --model $M/imp/SPL-b567-k32.onnx --code DA-SPL --split select | tail -1
python -m src.coco_retrieval table --ref A0 --codes A0 SPL-b567-k32 DA-A0 DA-SPL --split select --margin 2.0 --tag da
python -m src.da_gate gate2
echo "== 판단 ② 통과 → confirm·CIFAR test 한 번"
python -m src.coco_retrieval eval --model $M/clip_vitb16_image_fp32.onnx --code DA-A0 --split confirm --i-know-final | tail -1
python -m src.coco_retrieval eval --model $M/imp/SPL-b567-k32.onnx --code DA-SPL --split confirm --i-know-final | tail -1
python -m src.evaluate --model $M/clip_vitb16_image_fp32.onnx --code DA-A0 --quant-mode fp32 --eval-set test --i-know-final | tail -1
python -m src.evaluate --model $M/imp/SPL-b567-k32.onnx --code DA-SPL --quant-mode static --eval-set test --i-know-final | tail -1
python -m src.coco_retrieval table --ref A0 --codes A0 SPL-b567-k32 DA-A0 DA-SPL --split confirm --margin 2.0 --tag da
python -m src.coco_retrieval table --ref DA-A0 --codes DA-A0 DA-SPL --split confirm --margin 2.0 --tag da_quant
python -m src.da_gate decide
