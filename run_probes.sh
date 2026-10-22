#!/usr/bin/env bash
# P4 회복 탐침 + 캘리브레이션 도메인 A/B. 첫 비교표에서 의미 있는 손실이 확인됐을 때만 돌린다.
#
#   bash run_probes.sh            # 전부
#   bash run_probes.sh probes     # 회복 탐침만
#   bash run_probes.sh ab         # 캘리브레이션 A/B 만
#
# 정확도만 잰다. 속도는 정밀도 구성이 바뀌는 경우(층 제외)에만 의미가 있으므로 따로 잰다:
#   python -m src.benchmark --models A2=<경로> P-ln=<경로> ...
set -euo pipefail
cd "$(dirname "$0")"
THREADS=${THREADS:-4}
WHAT=${1:-all}
M=$(python -m src.common models-dir)
BASE=splits/calib_coco_64_s0.txt

one() {  # 코드 출력파일 [양자화 인자...]
  local code=$1 out="$M/$2"; shift 2
  [ -f "$out" ] || python -m src.quantize_ort --mode static --out "$out" "$@"
  [ -f "results/preds/$code-val.json" ] || python -m src.evaluate --model "$out" --code "$code" \
      --threads "$THREADS" --meta "${out%.onnx}.json" ${EVAL_ARGS:-}
}

if [ "$WHAT" = all ] || [ "$WHAT" = probes ]; then
  echo "== 회복 탐침 (기준: A2 = MinMax, per-tensor, COCO 64장 s0)"
  one P-pc      p_pc.onnx      --calib-list $BASE --per-channel
  one P-pc-rr   p_pc_rr.onnx   --calib-list $BASE --per-channel --reduce-range
  one P-pct     p_pct999.onnx  --calib-list $BASE --calib-method percentile --percentile 99.9
  one P-ent     p_entropy.onnx --calib-list $BASE --calib-method entropy
  one P-ln      p_ln.onnx      --calib-list $BASE --protect layernorm
  one P-add     p_add.onnx     --calib-list $BASE --protect residual_add
  one P-pe      p_pe.onnx      --calib-list $BASE --protect patch_embed
fi

if [ "$WHAT" = all ] || [ "$WHAT" = ab ]; then
  echo "== 캘리브레이션 도메인 A/B (A: COCO, B: CIFAR-100 train. 각 3묶음, 나머지 조건 동일)"
  # A 조건의 s0 는 A2 와 같은 모델이다.
  for s in 1 2; do one AB-coco-s$s ab_coco_s$s.onnx --calib-list splits/calib_coco_64_s$s.txt; done
  for s in 0 1 2; do one AB-cifar-s$s ab_cifar_s$s.onnx --calib-list splits/calib_cifar_64_s$s.txt; done
fi

python -m src.make_table --all
