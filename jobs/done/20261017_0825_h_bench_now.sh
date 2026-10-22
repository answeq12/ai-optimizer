# ORT 속도 측정 즉시 실행 (10/17 08:25, 사용자 요청 "지금 해"). 밤 대기(job d)가 2번 연속 깨어나지 못해 시간 조건을 뺌
# 부하(1분 평균) 2 미만을 최대 30분 기다린 뒤 측정. 측정 중에는 PC 사용 자제. 측정 시작·끝 부하를 로그에 남김
set -e
set -e
for i in $(seq 1 60); do l=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($l < 2.0)}" && break; sleep 30; done
source ~/venvs/halio/bin/activate
M=$(python -m src.common models-dir); R=$M/res
MODELS="A0=$M/clip_vitb16_image_fp32.onnx A1=$M/a1_dynamic.onnx S-c67-pc=$M/imp/c67_pc.onnx SPL-b567-k32=$M/imp/SPL-b567-k32.onnx A1-c67-pc=$M/imp/a1_c67_pc.onnx"
for S in 192 176 160; do
  [ -f $R/clip_vitb16_image_fp32_r$S.onnx ] && MODELS="$MODELS R$S-A0=$R/clip_vitb16_image_fp32_r$S.onnx"
  [ -f $R/r${S}_a1_c67_pc.onnx ] && MODELS="$MODELS R$S-A1-c67-pc=$R/r${S}_a1_c67_pc.onnx"
done
echo "== 측정 시작 $(date '+%F %T') / 부하 $(cut -d' ' -f1-3 /proc/loadavg)"; echo "$MODELS" | tr ' ' '\n'
python -m src.benchmark --threads 4 --out-csv results/latency_20261017_morning.csv --models $MODELS
echo "== 커버리지"; python -m src.coverage --threads 4 --models SPL-b567-k32="$M/imp/SPL-b567-k32.onnx" A1-c67-pc="$M/imp/a1_c67_pc.onnx" || echo "(커버리지 실패, 속도에는 영향 없음)"
python -m src.make_table --all > /dev/null
echo "== 끝 $(date '+%F %T') / 부하 $(cut -d' ' -f1-3 /proc/loadavg)"
grep -E "^\| (A0|A1|S-c67-pc|SPL-b567-k32|A1-c67-pc|R1[0-9]{2}-)" results/all_runs.md || true
