# 재현 패키지에 최종 구성(DA-SPL) 반영 후 점검 (약 2시간, 정확도만)
#  a) 적응 가중치를 repro/weights 로 복사 + sha256, EXPECTED.json 에 DA-A0/DA-SPL 추가
#  b) 빠른 재현: 새 데이터 폴더(/mnt/d/halio_quickcheck2)에서 repro_quick.sh → 저장 결과와 교차 확인 → EXPECTED_QUICK 에 DA 추가 → 재판정
#  c) 전체 재현: 새 저장 폴더(/mnt/d/halio_fullcheck, 데이터·가중치 캐시만 링크)에서 PFX=RPV SKIP_BENCH=1 reproduce.sh
set -eo pipefail
source ~/venvs/halio/bin/activate
mkdir -p repro/weights
cp /mnt/d/halio_data/models_da/da_tail_chosen.pt /mnt/d/halio_data/models_da/da_tail_chosen.json repro/weights/
(cd repro/weights && sha256sum da_tail_chosen.pt > SHA256SUMS && cat SHA256SUMS)
python - <<'PY'
import json
from pathlib import Path
p=Path("repro/EXPECTED.json"); E=json.loads(p.read_text())
for code in ("DA-A0","DA-SPL"):
    c=json.loads(Path(f"results/coco_ret/{code}-confirm.json").read_text()); t=json.loads(Path(f"results/preds/{code}-test.json").read_text())
    E["accuracy"][code]={"confirm":{k:round(c[k],6) for k in ("R@1","R@5","R@10")},"test_top1":round(t["top1_acc"],4)}
E["_설명"]+=". DA-* 는 10/20 최종 구성(도메인 적응 + SPL, docs/02 E33)"
p.write_text(json.dumps(E,indent=2,ensure_ascii=False)); print(json.dumps(E["accuracy"],ensure_ascii=False))
PY
echo "== b) 빠른 재현 (새 폴더)"
Q=/mnt/d/halio_quickcheck2; mkdir -p $Q/data
[ -e $Q/data/cifar-100-python ] || ln -s /mnt/d/halio_data/data/cifar-100-python $Q/data/cifar-100-python
[ -e $Q/cache ] || ln -s /mnt/d/halio_data/cache $Q/cache
HALIO_DATA_DIR=$Q bash repro_quick.sh || echo "(DA 기대값이 아직 없어 비교 단계 실패는 정상)"
python - <<'PY'
import json, os
import numpy as np
from pathlib import Path
os.environ.pop("HALIO_QUICK_N",None)
from src import coco_retrieval as R
E=json.loads(Path("repro/EXPECTED_QUICK.json").read_text())
files,img_of,caps=R.captions_for("confirm"); txt=np.load("/mnt/d/halio_data/models/coco_ret_text_confirm.npy"); m=img_of<500
for code in ("DA-A0","DA-SPL"):
    c=json.loads(Path(f"results/coco_ret/RPQ-{code}-confirm.json").read_text()); t=json.loads(Path(f"results/preds/RPQ-{code}-test2000-test.json").read_text())
    emb=np.load(f"results/coco_ret/{code}-confirm.npz")["emb"][:500]; r=R.ranks(emb,txt[m],img_of[m])
    z=np.load(f"results/preds/{code}-test.npz"); acc=float((z["preds"][:2000]==z["labels"][:2000]).mean())
    print(f"[교차] {code}: 빠른 재현 R@1 {c['R@1']:.4f} / 저장 결과로 계산 {(r<1).mean():.4f} | test2000 {t['top1_acc']:.4f} / {acc:.4f}")
    E["accuracy"][code]={"confirm":{k:c[k] for k in ("R@1","R@5","R@10")},"test_top1":t["top1_acc"]}
E["_설명"]+=". DA-* 는 10/20 최종 구성(도메인 적응 + SPL)"
Path("repro/EXPECTED_QUICK.json").write_text(json.dumps(E,indent=2,ensure_ascii=False)); print("EXPECTED_QUICK 갱신")
PY
python repro/check_expected.py --quick
echo "== c) 전체 재현 (새 저장 폴더, 정확도만)"
V=/mnt/d/halio_fullcheck; mkdir -p $V/data
for d in cifar-100-python coco; do [ -e $V/data/$d ] || ln -s /mnt/d/halio_data/data/$d $V/data/$d; done
[ -e $V/cache ] || ln -s /mnt/d/halio_data/cache $V/cache
PFX=RPV SKIP_BENCH=1 HALIO_DATA_DIR=$V bash reproduce.sh
