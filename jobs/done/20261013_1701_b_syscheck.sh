# 속도 측정 전 PC 상태 점검 (WSL 안에서)
echo "== 시각 $(date '+%F %T')"
echo "== 부하(1/5/15분): $(cut -d' ' -f1-3 /proc/loadavg)  / 논리코어 $(nproc)"
echo "== 메모리"; free -g | head -2
echo "== CPU 상위 프로세스"; ps -eo pid,pcpu,pmem,etime,comm --sort=-pcpu | head -8
echo "== python 프로세스"; ps -eo pid,pcpu,pmem,etime,args | grep -i [p]ython | cut -c1-150
echo "== GPU"; (nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null) || echo "nvidia-smi 없음"
