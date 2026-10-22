# E40 속도·메모리 실측 (약 20~30분). 다른 프로그램을 끄고 기다린다(부하 2.0 미만, 최대 30분 대기)
set -eo pipefail
source ~/venvs/halio/bin/activate
S=/mnt/d/halio_data
for i in $(seq 1 60); do l=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($l < 2.0)}" && break; sleep 30; done
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2) / 부하 $(cut -d' ' -f1-3 /proc/loadavg) / $(date '+%F %T')"
python -m src.benchmark --threads 4 --repeats 3 --out-csv results/latency_e40.csv --models \
  A0=$S/models/clip_vitb16_image_fp32.onnx A1=$S/models/a1_dynamic.onnx SPL-b567-k32=$S/models/imp/SPL-b567-k32.onnx \
  QAT-SPL=$S/models_qat_orig/imp/SPL-b567-k32.onnx DA-SPL=$S/models_da/imp/SPL-b567-k32.onnx
echo "끝 부하 $(cut -d' ' -f1-3 /proc/loadavg) / $(date '+%F %T')"
python - <<'PY'
import csv, statistics
from collections import defaultdict
R=defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open("results/latency_e40.csv")):
    for k in ("latency_p50_ms","latency_p95_ms","max_rss_mb","model_size_mb"):
        if r.get(k): R[r["config_code"]][k].append(float(r[k]))
a0=statistics.median(R["A0"]["latency_p50_ms"])
L=["# E40 속도·메모리 (PC00, 4스레드·배치 1, 3반복 중앙값)","","| 구성 | p50 ms | p95 ms | FP32 대비 | 최대 메모리 MB | 크기 MB |","|---|---|---|---|---|---|"]
for c in ("A0","A1","SPL-b567-k32","QAT-SPL","DA-SPL"):
    m=lambda k: statistics.median(R[c][k]) if R[c][k] else float("nan")
    L.append(f"| {c} | {m('latency_p50_ms'):.2f} | {m('latency_p95_ms'):.2f} | {a0/m('latency_p50_ms'):.2f}배 | {m('max_rss_mb'):.0f} | {m('model_size_mb'):.1f} |")
open("results/latency_e40.md","w").write("\n".join(L)+"\n"); print("\n".join(L))
PY
