# 작업 1 정합성: (1) 우리 ONNX/open_clip 텍스트 vs HF openai/clip-vit-base-patch16 (별도 venv, 파일럿 환경 안 건드림)
# (2) COCO-select 64장으로 fc2 활성값 통계 재측정 (측정 전용)
set -e
if [ ! -x ~/venvs/hfcheck/bin/python ]; then
  python3 -m venv ~/venvs/hfcheck || ~/venvs/halio/bin/python -m venv ~/venvs/hfcheck
  ~/venvs/hfcheck/bin/pip install -q --upgrade pip
  ~/venvs/hfcheck/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  ~/venvs/hfcheck/bin/pip install -q transformers onnxruntime pillow numpy
fi
set +e
~/venvs/hfcheck/bin/python -m src.coco_retrieval check || echo "!! HF 정합성 확인 실패 (아래 로그)"
source ~/venvs/halio/bin/activate
python -m src.coco_retrieval statslist
python -m src.act_stats --calib-list splits/stats_coco_select_64.txt --tag coco_select > /dev/null && \
python - <<'PY'
import json
for tag in ("", "_coco_select", "_cifar"):
    try: d = json.load(open(f"results/act_stats{tag}.json"))
    except FileNotFoundError: continue
    rs = sorted([r for r in d["rows"] if r["kind"] == "fc2" and r["block"] in (4,5,6,7,8)], key=lambda r: r["block"])
    print(d["calib_list"], " | ".join(f"b{r['block']}: max {r['absmax']:.0f} / p99.9 {r['p999']:.2f} / {r['bulk_levels']:.1f}칸 / hot {r['hot_channels']}" for r in rs))
PY
