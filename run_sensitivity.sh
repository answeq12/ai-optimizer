#!/usr/bin/env bash
# 레이어 민감도 탐침. A2m(정적 INT8, MinMax, per-tensor, COCO 64장 s0, 가중치 MatMul 49개)에서
# 한 묶음만 FP32 로 남겨 정확도가 얼마나 돌아오는지 잰다. 바뀌는 변수는 "제외한 노드" 하나다.
#
#   bash run_sensitivity.sh              # 1단계: 종류별 5 + 깊이별 3 = 8개 (약 40~50분)
#   KIND=fc2 bash run_sensitivity.sh block   # 2단계: 한 종류를 블록 12개로 쪼개서 (1단계 결과를 본 뒤에만)
#
# 정확도만 잰다(전 조건). 속도는 FP32 로 남긴 만큼 달라지므로 최종 후보만 따로 잰다.
# DFC 컴파일과 동시에 돌리지 말 것 (WSL 메모리 23.5GB).
set -euo pipefail
cd "$(dirname "$0")"
THREADS=${THREADS:-4}
WHAT=${1:-stage1}
M=$(python -m src.common models-dir)
BASE=splits/calib_coco_64_s0.txt
SD=results/sensitivity
mkdir -p "$SD" "$M/sens"

[ -f results/preds/A2m-val.npz ] || { echo "!! A2m 결과가 없습니다. 실행파일_bat/2-3_정적INT8_A2m_재측정.bat 을 먼저 돌리세요"; exit 1; }

echo "== 가중치 MatMul 49개 분류 (블록 x 종류)"
python -m src.sensitivity groups

one() {  # 코드 선택규칙
  local code=$1 spec=$2 out="$M/sens/$1.onnx"
  echo "== $code: $spec 만 FP32 로 남김"
  python -m src.sensitivity names "$spec" --code "$code" --out "$SD/$code.nodes.txt"
  [ -f "$out" ] || python -m src.quantize_ort --mode static --weight-matmul-only \
      --calib-list $BASE --exclude-nodes-file "$SD/$code.nodes.txt" --out "$out"
  [ -f "results/preds/$code-val.json" ] || python -m src.evaluate --model "$out" --code "$code" \
      --quant-mode static --threads "$THREADS" --meta "${out%.onnx}.json"
  # 다음 실행 전에 지금까지 결과를 표로 남긴다 (중간에 끊겨도 본 만큼은 남게)
  python -m src.sensitivity table > /dev/null
}

case "$WHAT" in
  stage1)
    echo "== 1단계-a: 종류별 (블록 12개에 걸쳐 같은 종류를 한꺼번에 FP32)"
    one S-qkv  kind:qkv
    one S-out  kind:out
    one S-fc1  kind:fc1
    one S-fc2  kind:fc2
    one S-proj proj
    echo "== 1단계-b: 깊이별 (블록 4개씩, 네 종류 전부 FP32)"
    one S-d0-3  depth:0-3
    one S-d4-7  depth:4-7
    one S-d8-11 depth:8-11
    ;;
  block)
    : "${KIND:?KIND=qkv|out|fc1|fc2 를 지정하세요}"
    echo "== 2단계: $KIND 를 블록별로 하나씩 FP32"
    for b in $(seq 0 11); do one "S-b$b-$KIND" "block:$b/$KIND"; done
    ;;
  combo)
    echo "== 3단계: 2단계에서 민감했던 fc2 (블록 7 > 6 > 5) 를 조합해 FP32 로 남김 (혼합 정밀도 후보)"
    one S-c67-fc2    "blocks:6,7/fc2"
    one S-c567-fc2   "blocks:5,6,7/fc2"
    one S-c4567-fc2  "blocks:4,5,6,7/fc2"
    ;;
  *) echo "알 수 없는 단계: $WHAT (stage1 / block / combo)"; exit 1 ;;
esac

python -m src.sensitivity table
echo "== 끝. 확인할 파일: results/sensitivity.md"
