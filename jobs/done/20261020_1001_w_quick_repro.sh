# (2/3) 빠른 재현 모드 만들기 + 점검. 정확도만. 약 20~30분
#  a) repro/quick/coco_files.txt (confirm 앞 500 + 캘리브 64), captions_quick.json(그 사진의 COCO 캡션) 생성
#  b) 새 데이터 폴더(/mnt/d/halio_quickcheck)에서 repro_quick.sh 실행. CIFAR·CLIP 가중치 캐시는 링크로 재사용(속도), COCO 사진은 실제로 한 장씩 받음
#  c) 그 결과로 repro/EXPECTED_QUICK.json 작성 + 저장된 전체 결과(npz)로 계산한 값과 교차 확인 → check_expected --quick 재실행
set -e
source ~/venvs/halio/bin/activate
python - <<'PY'
import json
from pathlib import Path
from src import common as C
conf=[l.strip() for l in open("splits/coco_ret_confirm.txt").read().split("\n")[1:] if l.strip()][:500]
cal=[l.strip() for l in open("splits/calib_coco_64_s0.txt").read().split("\n")[1:] if l.strip()]
names=conf+[c for c in cal if c not in conf]
Path("repro/quick").mkdir(parents=True, exist_ok=True)
Path("repro/quick/coco_files.txt").write_text("# confirm 앞 500장 + 캘리브 s0 64장 (COCO val2017)\n"+"\n".join(names)+"\n")
full=json.loads((C.DATA/"coco"/"annotations"/"captions_val2017.json").read_text())
keep={im["id"] for im in full["images"] if im["file_name"] in set(names)}
sub={"info":full.get("info"),"licenses":full.get("licenses"),
     "images":[im for im in full["images"] if im["id"] in keep],
     "annotations":[a for a in full["annotations"] if a["image_id"] in keep]}
Path("repro/quick/captions_quick.json").write_text(json.dumps(sub,ensure_ascii=False))
print("quick 목록", len(names), "장, 캡션", len(sub["annotations"]), "개")
PY
Q=/mnt/d/halio_quickcheck; mkdir -p $Q/data
[ -e $Q/data/cifar-100-python ] || ln -s /mnt/d/halio_data/data/cifar-100-python $Q/data/cifar-100-python
[ -e $Q/cache ] || ln -s /mnt/d/halio_data/cache $Q/cache
HALIO_DATA_DIR=$Q bash repro_quick.sh || echo "(첫 실행: 기대값 파일이 아직 없어 비교 단계 실패는 정상)"
python - <<'PY'
import json, os
import numpy as np
from pathlib import Path
res={}
for code in ("A0","SPL-b567-k32"):
    c=json.loads(Path(f"results/coco_ret/RPQ-{code}-confirm.json").read_text())
    t=json.loads(Path(f"results/preds/RPQ-{code}-test2000-test.json").read_text())
    res[code]={"confirm":{k:c[k] for k in ("R@1","R@5","R@10")},"test_top1":t["top1_acc"]}
# 교차 확인: 원래 전체 실행의 저장 임베딩으로 같은 부분집합 값을 다시 계산
os.environ.pop("HALIO_QUICK_N",None)
from src import coco_retrieval as R
files,img_of,caps=R.captions_for("confirm")
txt=np.load("/mnt/d/halio_data/models/coco_ret_text_confirm.npy")
m=img_of<500
for code in ("A0","SPL-b567-k32"):
    emb=np.load(f"results/coco_ret/{code}-confirm.npz")["emb"][:500]
    r=R.ranks(emb,txt[m],img_of[m])
    z=np.load(f"results/preds/{code}-test.npz"); acc=float((z["preds"][:2000]==z["labels"][:2000]).mean())
    print(f"[교차] {code}: 빠른 재현 R@1 {res[code]['confirm']['R@1']:.4f} / 저장 결과로 계산 {(r<1).mean():.4f} | test2000 {res[code]['test_top1']:.4f} / {acc:.4f}")
E={"_설명":"빠른 재현(repro_quick.sh) 기대값: COCO-confirm 앞 500장 갤러리 검색 R@K, CIFAR-100 test 앞 2,000장 top-1. PC00(i9-14900KF)에서 10/20 생성. 전체 결과(EXPECTED.json)와 같은 모델·같은 코드",
   "accuracy":res,"tolerance":{"match":0.002,"close":0.005}}
Path("repro/EXPECTED_QUICK.json").write_text(json.dumps(E,indent=2,ensure_ascii=False))
print("EXPECTED_QUICK 작성")
PY
python repro/check_expected.py --quick
