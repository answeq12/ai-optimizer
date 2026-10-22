# 혼합 정밀도 후보(A3b) 속도·메모리 측정. 같은 세션에서 A0·A1·A2m 도 다시 재서 나란히 비교
# 사용자 확인: 2026-10-06 22:15 다른 작업 없음
set -e
source ~/venvs/halio/bin/activate
THREADS=${THREADS:-4}
M=$(python -m src.common models-dir)
echo "== 측정 시작 $(date '+%F %T')  (다른 프로그램을 켜지 마세요)"
python -m src.benchmark --threads "$THREADS" --out-csv results/latency_a3b.csv \
  --models A0="$M/clip_vitb16_image_fp32.onnx" A1="$M/a1_dynamic.onnx" A2m="$M/a2m_minmax.onnx" \
           S-b7-fc2="$M/sens/S-b7-fc2.onnx" S-c67-fc2="$M/sens/S-c67-fc2.onnx" S-c567-fc2="$M/sens/S-c567-fc2.onnx"
echo "== 연산자 커버리지"
python -m src.coverage --threads "$THREADS" --models A1="$M/a1_dynamic.onnx" A2m="$M/a2m_minmax.onnx" \
  S-c67-fc2="$M/sens/S-c67-fc2.onnx" S-c567-fc2="$M/sens/S-c567-fc2.onnx"
python -m src.make_table --all > /dev/null
echo "== 끝 $(date '+%F %T'). 결과: results/latency_summary.json, results/all_runs.md, results/coverage.md"
