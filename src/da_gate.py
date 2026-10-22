"""E33 판단 ②와 교체 규칙 계산 (coco_retrieval table 과 같은 부트스트랩: 이미지 단위, 2000회, 시드 0).

  python -m src.da_gate gate2                 # select: DA-SPL vs DA-A0 R@1 차이 CI 하한 ≥ −2.0
  python -m src.da_gate decide                # confirm: DA-SPL vs A0 CI 하한 > 0 그리고 CIFAR test −1.0%p 이내
"""
from __future__ import annotations

import json
import sys

import numpy as np

from . import common as C
from .coco_retrieval import OUT, per_image


def diff_ci(code, ref, split, boot=2000):
    a, b = np.load(OUT / f"{code}-{split}.npz"), np.load(OUT / f"{ref}-{split}.npz")
    img_of = b["img_of"]; assert np.array_equal(a["img_of"], img_of)
    n = int(img_of.max() + 1)
    d = per_image(a["ranks"], img_of, n, 1) - per_image(b["ranks"], img_of, n, 1)
    idx = np.random.default_rng(0).integers(0, n, size=(boot, n))
    lo, hi = np.percentile(d[idx].mean(axis=1) * 100, [2.5, 97.5])
    return float(d.mean() * 100), float(lo), float(hi)


def main():
    what = sys.argv[1]
    if what == "gate2":
        m, lo, hi = diff_ci("DA-SPL", "DA-A0", "select")
        ok = lo >= -2.0
        res = {"gate2": {"diff": m, "ci": [lo, hi], "pass": ok}}
        print(f"[da-gate] 판단 ② select DA-SPL − DA-A0 R@1 {m:+.2f} [{lo:+.2f}, {hi:+.2f}] → {'통과' if ok else '미통과'}")
    else:
        m, lo, hi = diff_ci("DA-SPL", "A0", "confirm")
        acc = lambda c: json.loads((C.RESULTS / "preds" / f"{c}-test.json").read_text())["top1_acc"] * 100
        ct = acc("DA-SPL") - acc("A0")
        ok = lo > 0 and ct >= -1.0
        res = {"decide": {"confirm_diff_vs_A0": m, "ci": [lo, hi], "cifar_test_diff_vs_A0": ct, "replace": ok}}
        print(f"[da-gate] 교체 규칙: confirm DA-SPL − A0 R@1 {m:+.2f} [{lo:+.2f}, {hi:+.2f}], CIFAR test {ct:+.2f}%p → "
              f"{'제출 구성 교체' if ok else '현 구성 유지(DA 는 추가 탐색으로 보고)'}")
    p = C.RESULTS / "da_decision.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(res); p.write_text(json.dumps(cur, indent=2, ensure_ascii=False))
    if what == "gate2" and not res["gate2"]["pass"]:
        sys.exit(4)


if __name__ == "__main__":
    main()
