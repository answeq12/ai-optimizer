# 밤 속도 측정 (한 세션): ORT 후보 5개 + 해상도 축소 모델. 10/16 투입
# 10/14·10/15 밤에 'sleep 으로 03:00 까지 대기'가 PC 절전 중 시간이 멈춰 깨어나지 못함 -> 1분마다 시계를 확인하는 방식으로 바꿈
# 시작 조건: 01:00~06:00 사이 + 부하(1분 평균) 2 미만. 약 40~50분
set -e
while true; do h=$(date +%H); if [ "$h" -ge 1 ] && [ "$h" -lt 6 ]; then break; fi; sleep 60; done
for i in $(seq 1 60); do l=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($l < 2.0)}" && break; sleep 30; done
source ~/venvs/halio/bin/activate
M=$(python -m src.common models-dir); R=$M/res
MODELS="A0=$M/clip_vitb16_image_fp32.onnx A1=$M/a1_dynamic.onnx S-c67-pc=$M/imp/c67_pc.onnx SPL-b567-k32=$M/imp/SPL-b567-k32.onnx A1-c67-pc=$M/imp/a1_c67_pc.onnx"
for S in 192 176 160; do
  [ -f $R/clip_vitb16_image_fp32_r$S.onnx ] && MODELS="$MODELS R$S-A0=$R/clip_vitb16_image_fp32_r$S.onnx"
  [ -f $R/r${S}_a1_c67_pc.onnx ] && MODELS="$MODELS R$S-A1-c67-pc=$R/r${S}_a1_c67_pc.onnx"
done
echo "== 측정 시작 $(date '+%F %T') / 부하 $(cut -d' ' -f1-3 /proc/loadavg)"; echo "$MODELS" | tr ' ' '\n'
python -m src.benchmark --threads 4 --out-csv results/latency_night_20261017.csv --models $MODELS
echo "== 커버리지"; python -m src.coverage --threads 4 --models SPL-b567-k32="$M/imp/SPL-b567-k32.onnx" A1-c67-pc="$M/imp/a1_c67_pc.onnx" || echo "(커버리지 실패, 속도에는 영향 없음)"
python -m src.make_table --all > /dev/null
echo "== 끝 $(date '+%F %T') / 부하 $(cut -d' ' -f1-3 /proc/loadavg)"
grep -E "^\| (A0|A1|S-c67-pc|SPL-b567-k32|A1-c67-pc|R1[0-9]{2}-)" results/all_runs.md || true
