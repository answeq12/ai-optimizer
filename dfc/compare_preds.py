"""Hailo 에뮬레이터 평가 결과(logs/dfc/preds_*.npz)를 같은 이미지끼리 짝지어 비교한다.

  python -m dfc.compare_preds logs/dfc/preds.coco64_base_n2000.npz logs/dfc/preds.cifar64_base_n2000.npz ...
첫 번째 파일이 기준. 결과는 화면과 logs/dfc/compare_preds.md 에 남긴다.
"""
from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np


def mcnemar(a: np.ndarray, b: np.ndarray):
    x, y = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n, k = x + y, min(x, y)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return x, y, p


def main() -> None:
    files = [Path(f) for f in sys.argv[1:]]
    data = {f.name: np.load(f) for f in files}
    names = list(data)
    base = data[names[0]]
    for n in names[1:]:
        if not np.array_equal(data[n]["labels"], base["labels"]):
            sys.exit(f"{n}: 기준과 이미지 목록이 다릅니다. 짝지은 비교 불가")
    fp = base["fp"] == base["labels"]
    L = [f"# Hailo 에뮬레이터 짝지은 비교 ({len(fp)}장, 같은 이미지)", "",
         f"FP(H0) top-1: {fp.mean():.2%} (기준 파일 {names[0]})", "",
         "| 구성 | 양자화 top-1 | 표준오차 | FP 대비 | FP 일치율 | 평균 코사인 | 기준만 정답 / 이것만 정답 | McNemar p (vs 기준) |",
         "|---|---|---|---|---|---|---|---|"]
    bq = base["q"] == base["labels"]
    for n in names:
        d = data[n]
        q = d["q"] == d["labels"]
        se = 100 * np.sqrt(q.mean() * (1 - q.mean()) / len(q))
        x, y, p = mcnemar(bq, q)
        L.append(f"| {n} | {q.mean():.2%} | ±{se:.2f}%p | {(q.mean() - fp.mean()) * 100:+.2f}%p | "
                 f"{(d['q'] == d['fp']).mean():.2%} | {d['cos'].mean():.4f} | {x} / {y} | {p:.2g} |")
    L += ["", "- Hailo DFC 에뮬레이터 수치다. 실기기 수치가 아니며 ORT 결과와 직접 비교하지 않는다"]
    text = "\n".join(L) + "\n"
    Path("logs/dfc/compare_preds.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
