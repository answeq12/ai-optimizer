# (3/3) 일반화 확인 (사후 추가 탐색): 우리 방법을 CLIP ViT-B/32 에 그대로 적용. 약 1~1.5시간, 정확도만
#  ① 캘리브 1회 활성값 지표(fc2 입력 bulk_levels < 20)로 보호할 블록 예측 → ② 블록별 fc2 민감도 실측(12회)으로 맞는지 확인
#  ③ 예측 블록에 이상치 채널 분리(k=32) 적용 → CIFAR 검증 + COCO-select R@1. confirm 은 쓰지 않음
#  모델 파일은 /mnt/d/halio_data/models_b32 (B/16 파일과 분리). 파일명 clip_vitb16_* 는 코드 공용 이름일 뿐 내용은 B/32
set -e
source ~/venvs/halio/bin/activate
export HALIO_MODEL=ViT-B-32 HALIO_MODELS_SUBDIR=models_b32
M=$(python -m src.common models-dir); mkdir -p $M/imp $M/sens; echo "모델 폴더 $M"
CAL=splits/calib_coco_64_s0.txt
cp logs/p1_parity.json logs/p1_parity.b16.keep
[ -f $M/clip_vitb16_image_fp32.onnx ] || python -m src.export_onnx
cp logs/p1_parity.json logs/p1_parity_b32.json; cp logs/p1_parity.b16.keep logs/p1_parity.json
python - <<'PY'
import json
from src import common as C
from src.quantize_ort import preprocess_model
from src.sensitivity import classify
b16=[r["name"] for r in json.load(open("results/sensitivity/groups.json"))["nodes"]]
b32=[r["name"] for r in classify()]
print("노드 이름 일치(B/16 groups.json 재사용 가능):", b16==b32, len(b32))
assert b16==b32, "노드 이름이 다르면 중단"
PY
python -m src.act_stats --tag b32 > /dev/null
BL=$(python - <<'PY'
import json
rs=sorted([r for r in json.load(open("results/act_stats_b32.json"))["rows"] if r["kind"]=="fc2"],key=lambda r:r["block"])
for r in rs: print(f"   b{r['block']}: max {r['absmax']:.1f} / {r['bulk_levels']:.1f}칸 / hot {r['hot_channels']}", file=__import__('sys').stderr)
low=[r["block"] for r in rs if r["bulk_levels"]<20] or [r["block"] for r in sorted(rs,key=lambda r:r["bulk_levels"])[:3]]
print(",".join(map(str,sorted(low))))
PY
)
echo "== 활성값 지표가 고른 블록: $BL"
ev() { [ -f results/preds/$1-val.json ] || python -m src.evaluate --model "$2" --code "$1" --quant-mode "$3" --eval-set val > /dev/null; python -c "import json;print('   $1 CIFAR 검증', round(json.load(open('results/preds/$1-val.json'))['top1_acc']*100,2))"; }
evc() { [ -f results/coco_ret/$1-select.json ] || python -m src.coco_retrieval eval --model "$2" --code "$1" --split select | tail -1; }
python -m src.coco_retrieval text
ev B32-A0 $M/clip_vitb16_image_fp32.onnx fp32; evc B32-A0 $M/clip_vitb16_image_fp32.onnx
[ -f $M/a2m.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --calib-list $CAL --out $M/a2m.onnx
ev B32-A2m $M/a2m.onnx static; evc B32-A2m $M/a2m.onnx
[ -f $M/a2m_pc.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --calib-list $CAL --out $M/a2m_pc.onnx
ev B32-A2m-pc $M/a2m_pc.onnx static; evc B32-A2m-pc $M/a2m_pc.onnx
echo "== 블록별 fc2 민감도 (한 블록 fc2 만 FP32, 나머지 A2m 과 같게)"
for b in $(seq 0 11); do
  c=B32-S-b$b-fc2; python -m src.sensitivity names "block:$b/fc2" --code $c --out results/sensitivity/$c.nodes.txt > /dev/null
  [ -f $M/sens/$c.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --calib-list $CAL --exclude-nodes-file results/sensitivity/$c.nodes.txt --out $M/sens/$c.onnx > /dev/null
  ev $c $M/sens/$c.onnx static
done
echo "== 이상치 채널 분리 (블록 $BL, k=32)"
F=$M/imp/split_SPL.fp32.onnx
[ -f $F ] || python -m src.split_outliers --blocks $BL --k 32 --calib-list $CAL --out $F
[ -f $M/imp/SPL.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --src $F --exclude-nodes-file $F.hot_nodes.txt --calib-list $CAL --out $M/imp/SPL.onnx
ev B32-SPL $M/imp/SPL.onnx static; evc B32-SPL $M/imp/SPL.onnx
python -m src.coco_retrieval table --ref B32-A0 --codes B32-A0 B32-A2m B32-A2m-pc B32-SPL --split select --margin 2.0 --tag b32
BL="$BL" python - <<'PY'
import json, os
from pathlib import Path
acc=lambda c: json.load(open(f"results/preds/{c}-val.json"))["top1_acc"]*100
a0,a2=acc("B32-A0"),acc("B32-A2m"); gap=a0-a2
act={r["block"]:r for r in json.load(open("results/act_stats_b32.json"))["rows"] if r["kind"]=="fc2"}
rows=[(b,acc(f"B32-S-b{b}-fc2")) for b in range(12)]
rec={b:(a-a2)/gap*100 if gap>0 else float("nan") for b,a in rows}
top3=sorted(rec,key=lambda b:-rec[b])[:3]; pred=[int(x) for x in os.environ["BL"].split(",")]
L=["# 일반화 확인: 같은 방법을 CLIP ViT-B/32 에 적용 (사후 추가 탐색)","",
   f"CIFAR-100 검증 top-1: FP32 {a0:.2f}% / 보호 없는 정적 INT8(A2m) {a2:.2f}% (손실 {gap:.2f}%p) / A2m+채널별 {acc('B32-A2m-pc'):.2f}% / **이상치 채널 분리(블록 {os.environ['BL']}, k=32) {acc('B32-SPL'):.2f}%**","",
   "| 블록 | fc2 입력 최대 | 8비트 눈금 사용(칸) | 이 블록 fc2 만 FP32 일 때 | 손실 회복률 |","|---|---|---|---|---|"]
for b,a in rows:
    L.append(f"| b{b}{' ←예측' if b in pred else ''} | {act[b]['absmax']:.1f} | {act[b]['bulk_levels']:.1f} | {a:.2f}% | {rec[b]:.0f}% |")
L+=["",f"- 활성값 지표(캘리브 1회)가 고른 블록: {pred} / 민감도 실측 상위 3블록: {top3} / 겹침 {sorted(set(pred)&set(top3))}",
    "- COCO-select 검색 결과는 results/coco_ret_select_b32.md","- B/16 과 같은 코드·같은 캘리브(COCO 64장)·같은 설정. 사후 추가 탐색이며 제출 구성과 무관"]
Path("results/b32_generalization.md").write_text("\n".join(L)+"\n"); print("\n".join(L))
PY
