#!/usr/bin/env bash
# A2 재정의. 정적 INT8 을 동적(A1)과 같은 대상(가중치 MatMul)으로 제한해 다시 잰다.
#
# 배경: 원래 A2 는 ORT 기본값으로 Softmax·Sigmoid·Mul·Add 까지 전부 INT8 로 바꿔 top-1 2.09% 로 붕괴했다.
# A1(동적)은 가중치 MatMul 49개만 양자화했다. 대상이 달라서 두 결과는 비교할 수 없었다.
# 원래 A2 와 탐침 결과는 지우지 않는다. "효과 없던 시도" 기록으로 보고서에 남긴다.
#
#   bash run_a2fix.sh
# DFC 컴파일과 동시에 돌리지 말 것 (WSL 메모리 23.5GB 를 나눠 쓰면 둘 다 죽을 수 있다).
set -euo pipefail
cd "$(dirname "$0")"
THREADS=${THREADS:-4}
M=$(python -m src.common models-dir)
BASE=splits/calib_coco_64_s0.txt

one() {  # 코드 출력파일 [양자화 인자...]
  local code=$1 out="$M/$2"; shift 2
  [ -f "$out" ] || python -m src.quantize_ort --mode static --weight-matmul-only --out "$out" "$@"
  [ -f "results/preds/$code-val.json" ] || python -m src.evaluate --model "$out" --code "$code" \
      --threads "$THREADS" --meta "${out%.onnx}.json" ${EVAL_ARGS:-}
}

echo "== A2m: 정적 INT8, MinMax, per-tensor, COCO 64장 s0, 가중치 MatMul 만 (A1 과 같은 대상)"
one A2m      a2m_minmax.onnx  --calib-list $BASE
echo "== A2m 위에서 탐침 두 개 (대상을 맞춘 뒤에야 의미가 있다)"
one A2m-pc   a2m_pc.onnx      --calib-list $BASE --per-channel
one A2m-pct  a2m_pct999.onnx  --calib-list $BASE --calib-method percentile --percentile 99.9

echo "== 연산자 커버리지 (기대: A2m 의 행렬곱 정수/실수 = A1 과 같은 49 / 25)"
python -m src.coverage --threads "$THREADS" --models A1="$M/a1_dynamic.onnx" A2m="$M/a2m_minmax.onnx"

echo "== 속도·메모리 (같은 세션에서 A0·A1 도 다시 재서 나란히 비교)"
python -m src.benchmark --threads "$THREADS" --out-csv results/latency_a2fix.csv \
  --models A0="$M/clip_vitb16_image_fp32.onnx" A1="$M/a1_dynamic.onnx" \
           A2m="$M/a2m_minmax.onnx" A2m-pc="$M/a2m_pc.onnx" A2m-pct="$M/a2m_pct999.onnx"

python -m src.make_table --all
echo "== 끝. 확인할 파일: results/all_runs.md, results/coverage.md, results/latency_a2fix.csv"
