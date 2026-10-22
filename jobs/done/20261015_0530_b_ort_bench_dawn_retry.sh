# 재투입(10/15 05:30): 10/14 밤 WSL 이 멈춰 측정이 안 됨. 다음 03:00 에 실행
# ORT 마진 통과 후보 속도·메모리 (사용자 요청 10/14: 새벽에 측정). 앞 작업이 끝난 뒤 03:00 이후에 시작, 부하가 낮을 때만
# A0·A1·S-c67-pc 를 같은 세션에서 다시 재서 SPL-b567-k32·A1-c67-pc 와 나란히 비교. 약 30~40분
set -e
now=$(date +%s); tgt=$(date -d "03:00" +%s); [ $tgt -le $now ] && tgt=$(date -d "tomorrow 03:00" +%s)   # 항상 다음 03:00 (재투입 10/15)
if [ $tgt -gt $now ]; then echo "== $(date '+%F %T') : 03:00 까지 대기 ($(( (tgt-now)/60 ))분)"; sleep $((tgt-now)); fi
for i in $(seq 1 60); do      # 부하가 1분 평균 2 미만이 될 때까지 최대 30분 대기
  l=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($l < 2.0)}" && break; echo "부하 $l, 30초 대기"; sleep 30
done
source ~/venvs/halio/bin/activate
THREADS=${THREADS:-4}
M=$(python -m src.common models-dir)
ls -la "$M/imp/SPL-b567-k32.onnx" "$M/imp/a1_c67_pc.onnx" "$M/imp/c67_pc.onnx"
echo "== 측정 시작 $(date '+%F %T') / 부하 $(cut -d' ' -f1-3 /proc/loadavg)"
python -m src.benchmark --threads "$THREADS" --out-csv results/latency_candidates_20261015.csv \
  --models A0="$M/clip_vitb16_image_fp32.onnx" A1="$M/a1_dynamic.onnx" \
           S-c67-pc="$M/imp/c67_pc.onnx" SPL-b567-k32="$M/imp/SPL-b567-k32.onnx" A1-c67-pc="$M/imp/a1_c67_pc.onnx"
echo "== 연산자 커버리지"
python -m src.coverage --threads "$THREADS" --models SPL-b567-k32="$M/imp/SPL-b567-k32.onnx" A1-c67-pc="$M/imp/a1_c67_pc.onnx" || echo "(커버리지 실패, 속도 결과에는 영향 없음)"
python -m src.make_table --all > /dev/null
echo "== 끝 $(date '+%F %T') / 부하 $(cut -d' ' -f1-3 /proc/loadavg)"
grep -E "^\| (A0|A1|S-c67-pc|SPL-b567-k32|A1-c67-pc) " results/all_runs.md
