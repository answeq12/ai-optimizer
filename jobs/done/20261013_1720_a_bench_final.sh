# 최종 후보 속도·메모리 측정 (사용자 확인: 2026-10-13 17:20 다른 작업 없음). 같은 세션에서 나란히. 약 35분
set -e
source ~/venvs/halio/bin/activate
THREADS=${THREADS:-4}
M=$(python -m src.common models-dir)
echo "== 측정 시작 $(date '+%F %T') / 부하 $(cut -d' ' -f1-3 /proc/loadavg)"
python -m src.benchmark --threads "$THREADS" --out-csv results/latency_final.csv \
  --models A0="$M/clip_vitb16_image_fp32.onnx" A1="$M/a1_dynamic.onnx" \
           S-c67-fc2="$M/sens/S-c67-fc2.onnx" S-c67-pc="$M/imp/c67_pc.onnx" S-c67-pc-qop="$M/imp/c67_pc_qop.onnx" \
           I4-b32="$M/imp/i4_b32.onnx" I4-b32-acc4="$M/imp/i4_b32_acc4.onnx"
echo "== 연산자 커버리지"
python -m src.coverage --threads "$THREADS" --models S-c67-pc="$M/imp/c67_pc.onnx" S-c67-pc-qop="$M/imp/c67_pc_qop.onnx" || echo "(커버리지 실패, 속도 결과에는 영향 없음)"
python -m src.make_table --all > /dev/null
echo "== 끝 $(date '+%F %T') / 부하 $(cut -d' ' -f1-3 /proc/loadavg)"
grep -E "^\| (A0|A1|S-c67-fc2|S-c67-pc|S-c67-pc-qop|I4-b32|I4-b32-acc4) " results/all_runs.md
