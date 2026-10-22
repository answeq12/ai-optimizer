# (1/3) 스레드·배치별 속도 (사후 측정, 보고서 6.5 보강). PC 조용할 때 — 사용자 요청으로 실행. 약 10분
set -e
source ~/venvs/halio/bin/activate
for i in $(seq 1 60); do l=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($l < 2.0)}" && break; sleep 30; done
echo "부하 $(cut -d' ' -f1-3 /proc/loadavg)"
python -m src.scaling_bench
