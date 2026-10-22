# 5개 구성 짝지은 비교 + 컨텍스트 수 -> logs/dfc/compare_preds.md
source ~/venvs/hailo/bin/activate
python -m dfc.compare_preds logs/dfc/preds.coco64_base_n2000.npz logs/dfc/preds.cifar64_base_n2000.npz \
  logs/dfc/preds.coco64_fc2b567_n2000.npz logs/dfc/preds.coco64_fc2b67_n2000.npz logs/dfc/preds.cifar64_fc2b567_n2000.npz
echo "== 컨텍스트 수 (컴파일 로그의 context_N 개수)"
for f in jobs/logs/*hailo_fc2*.log jobs/logs/*cifar_fc2*.log logs/dfc/setup_20261006_153317.log logs/dfc/setup_20261006_164810.log; do
  [ -f "$f" ] && echo "$(grep -c 'Compiling kernels of' "$f")  $f"
done | tee -a logs/dfc/compare_preds.md
