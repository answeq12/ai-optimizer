# 사진 색인·검색 실측 (10/19, 사후 측정). PC 조용할 때만 — 사용자 확인 받음 ("재줘"). 약 15~25분
#  색인: 1,000장 x 3구성(A0/A1/SPL) x 3반복, 검색: 질의 300개 + 색인 크기 1천/1만/10만
set -e
source ~/venvs/halio/bin/activate
echo "== 부하 낮아질 때까지 대기 (최대 30분)"
for i in $(seq 1 60); do l=$(cut -d' ' -f1 /proc/loadavg); awk "BEGIN{exit !($l < 2.0)}" && break; sleep 30; done
echo "   부하 $(cut -d' ' -f1-3 /proc/loadavg)"
echo "== 1) 짧은 점검 (30장, 1회)"
python -m src.index_bench run --n 30 --reps 1 --q 30 > /dev/null && echo "   점검 통과"
echo "== 2) 본 측정"
python -m src.index_bench run
