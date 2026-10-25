# 빠른 재현 기대값 DA5 갱신 + 교차 확인(전체 실행 저장 임베딩의 앞 500장으로 같은 값이 나오는지)
set -eo pipefail
source ~/venvs/halio/bin/activate
python - <<'PY'
import json, glob
import numpy as np
from pathlib import Path
from src.coco_retrieval import per_image, ranks
E = json.loads(Path("repro/EXPECTED_QUICK.json").read_text())
acc = {k: v for k, v in E["accuracy"].items() if not k.startswith("DA-")}
for c in ("DA5-A0", "DA5-SPL"):
    q = json.loads(Path(f"results/coco_ret/RPQ-{c}-confirm.json").read_text())
    t = json.loads(Path(f"results/preds/RPQ-{c}-test2000-test.json").read_text())
    acc[c] = {"confirm": {k: round(q[k], 6) for k in ("R@1", "R@5", "R@10")}, "test_top1": t["top1_acc"]}
    # 교차 확인: 원 실험 전체 confirm 임베딩 앞 500장 + 적응 텍스트(앞 500장 캡션)
    d = np.load(f"results/coco_ret/{c}-confirm.npz"); keep = d["img_of"] < 500
    txt = np.load("/mnt/d/halio_data/models_da5/coco_ret_text_confirm.npy")[keep]
    r = ranks(d["emb"][:500].astype(np.float32), txt, d["img_of"][keep])
    cap = {k: float((r < kk).mean()) for k, kk in (("R@1", 1), ("R@5", 5), ("R@10", 10))}
    print(c, "quick", acc[c]["confirm"], "| 전체 임베딩 앞 500장", {k: round(v, 6) for k, v in cap.items()})
E["accuracy"] = acc
E["_설명"] = "빠른 재현(repro_quick.sh) 기대값: COCO-confirm 앞 500장 갤러리 검색 R@K, CIFAR-100 test 앞 2,000장 top-1. PC00(i9-14900KF). DA5-* 는 10/25 최종 구성(도메인 적응 COCO 11.8만 + CC3M 10만, + SPL), 빈 폴더 빠른 재현으로 생성하고 전체 실행 임베딩 앞 500장과 교차 확인"
Path("repro/EXPECTED_QUICK.json").write_text(json.dumps(E, indent=2, ensure_ascii=False))
PY
python repro/check_expected.py --quick
