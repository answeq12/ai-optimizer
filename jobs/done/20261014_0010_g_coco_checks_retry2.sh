# 정합성 확인 재시도 2 (10/14): HF 투영 직접 계산, stats_ 목록은 측정 전용으로 가드 예외
source ~/venvs/halio/bin/activate
D=~/pydeps_hf
[ -d $D/transformers ] || pip install -q --target $D --no-deps transformers tokenizers
PYTHONPATH=$D python -m src.coco_retrieval check || echo "!! HF 정합성 확인 실패"
python -m src.coco_retrieval statslist
python -m src.act_stats --calib-list splits/stats_coco_select_64.txt --tag coco_select > /dev/null
python - <<'PY'
import json
for tag in ("", "_coco_select", "_cifar"):
    try: d = json.load(open(f"results/act_stats{tag}.json"))
    except FileNotFoundError: continue
    rs = sorted([r for r in d["rows"] if r["kind"] == "fc2" and r["block"] in (4,5,6,7,8)], key=lambda r: r["block"])
    print(d["calib_list"], " | ".join(f"b{r['block']}: max {r['absmax']:.0f} / p99.9 {r['p999']:.2f} / {r['bulk_levels']:.1f}칸 / hot {r['hot_channels']}" for r in rs))
PY
