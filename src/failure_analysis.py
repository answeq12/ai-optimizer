"""보고서 6.6 '어떤 입력에서 특히 취약했나': 양자화 손실을 입력 종류별로 나눠 본다. 사후 분석 (선택·규칙과 무관)

  python -m src.failure_analysis --split confirm     -> results/failure_analysis_confirm.md

- 이미 저장된 검색 순위(results/coco_ret/<code>-<split>.npz)만 쓴다. 모델을 다시 돌리지 않는다
- 비교 쌍: A0→SPL-b567-k32(ORT 최종), A0→A2m(보호 없는 정적 INT8, 패턴이 더 크게 보임),
           H-FP→H-splf-b7-k32(Hailo 동결), H-FP→H-official(Hailo 공식)
- 묶음: 캡션 특징(숫자·색·글자·위치 표현, 길이) / 사진 특징(COCO instances: 객체 수, 가장 큰 객체 면적 비율, 그 상위 범주)
- 묶음별 R@1 차이와 95% 범위 = 그 묶음의 이미지 단위 bootstrap 1000회. 묶음이 작으면 범위가 넓다 (n 함께 표기)
"""
from __future__ import annotations
import argparse, json, re, zipfile
import numpy as np
from . import common as C
from . import coco_retrieval as R

PAIRS = [("A0", "SPL-b567-k32", "ORT 최종"), ("A0", "A2m", "ORT 보호 없는 정적 INT8"),
         ("H-FP", "H-splf-b7-k32", "Hailo 동결"), ("H-FP", "H-official", "Hailo 공식")]
NUM = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|several|many|couple|pair|\d+)\b")
COLOR = re.compile(r"\b(red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey|silver|gold)\b")
TEXTW = re.compile(r"\b(sign|signs|text|written|says|reads|words|letters|logo|number|numbers)\b")
SPATIAL = re.compile(r"\b(left|right|behind|front|top|under|above|below|next to|beside|between|on top)\b")


def instances() -> dict:
    p = R.ANN_DIR / "instances_val2017.json"
    if not p.exists():
        z = C.DATA / "coco" / "annotations_trainval2017.zip"
        if not z.exists():
            from .download_data import _download
            print("[fail] COCO instances 다운로드 (약 241MB)"); _download(R.ANN_URL, z)
        with zipfile.ZipFile(z) as zf:
            p.write_bytes(zf.read("annotations/instances_val2017.json"))
    return json.loads(p.read_text())


def image_groups(files: list[str]) -> dict[str, np.ndarray]:
    ins = instances()
    idof = {im["file_name"]: im["id"] for im in ins["images"]}
    area = {im["id"]: im["width"] * im["height"] for im in ins["images"]}
    sup = {c["id"]: c["supercategory"] for c in ins["categories"]}
    objs: dict[int, list] = {}
    for a in ins["annotations"]:
        objs.setdefault(a["image_id"], []).append(a)
    n_obj, big, cat = [], [], []
    for f in files:
        i = idof[f]; os_ = objs.get(i, [])
        n_obj.append(len(os_))
        if os_:
            m = max(os_, key=lambda a: a["area"]); big.append(m["area"] / area[i]); cat.append(sup[m["category_id"]])
        else:
            big.append(0.0); cat.append("(객체 없음)")
    n_obj, big, cat = np.array(n_obj), np.array(big), np.array(cat)
    g = {"객체 0~1개": n_obj <= 1, "객체 2~5개": (n_obj >= 2) & (n_obj <= 5), "객체 6~15개": (n_obj >= 6) & (n_obj <= 15),
         "객체 16개 이상": n_obj >= 16,
         "가장 큰 객체 < 사진의 10%": big < 0.10, "가장 큰 객체 10~40%": (big >= 0.10) & (big < 0.40),
         "가장 큰 객체 ≥ 40%": big >= 0.40}
    for c in sorted(set(cat)):
        g[f"주 객체 범주: {c}"] = cat == c
    return g


def caption_groups(caps: list[str]) -> dict[str, np.ndarray]:
    low = [c.lower() for c in caps]
    L = np.array([len(c.split()) for c in caps])
    return {"캡션: 숫자·개수 표현": np.array([bool(NUM.search(c)) for c in low]),
            "캡션: 색 표현": np.array([bool(COLOR.search(c)) for c in low]),
            "캡션: 글자·표지판 표현": np.array([bool(TEXTW.search(c)) for c in low]),
            "캡션: 위치 관계 표현": np.array([bool(SPATIAL.search(c)) for c in low]),
            "캡션: 짧음(≤9단어)": L <= 9, "캡션: 김(≥13단어)": L >= 13, "캡션: 전체": np.ones(len(caps), bool)}


def boot_ci(d: np.ndarray, rng, n=1000):
    if len(d) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(d), (n, len(d)))
    return tuple(np.percentile(d[idx].mean(axis=1) * 100, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--split", default="confirm"); a = ap.parse_args()
    files, img_of, caps = R.captions_for(a.split); nimg = len(files)
    ig, cg = image_groups(files), caption_groups(caps)
    rng = np.random.default_rng(2026)
    L = [f"# 입력 종류별 양자화 손실 (COCO-{a.split}, 사후 분석)", "",
         f"생성 {C.now()}. 저장된 검색 순위만 사용. R@1 차이 = (양자화 − FP32) %p, 95% 범위 = 묶음 안 이미지 단위 bootstrap 1000회", "",
         "- 캡션 묶음: 그 표현이 들어간 캡션만 질의로 쓴 R@1 (갤러리는 전체 그대로)",
         "- 사진 묶음: 정답 사진이 그 묶음인 캡션만. 범주는 COCO instances 의 가장 큰 객체 기준", ""]
    out = {}
    for ref, cand, label in PAIRS:
        pr, pc = R.OUT / f"{ref}-{a.split}.npz", R.OUT / f"{cand}-{a.split}.npz"
        if not (pr.exists() and pc.exists()):
            L += [f"## {label}: {cand} vs {ref} — 결과 파일 없음, 건너뜀", ""]; continue
        hr = (np.load(pr)["ranks"] < 1).astype(float); hc = (np.load(pc)["ranks"] < 1).astype(float)
        L += [f"## {label}: {cand} vs {ref}", "", "| 묶음 | 캡션 수 | 사진 수 | FP32 R@1 | 양자화 R@1 | 차이 %p [95% 범위] |", "|---|---|---|---|---|---|"]
        rows = []
        for name, cmask in list(cg.items()) + [(k, v[img_of]) for k, v in ig.items()]:
            if cmask.sum() < 30:
                continue
            imgs = np.unique(img_of[cmask])
            # 이미지별: 이 묶음에 속한 캡션들의 hit 차이 평균
            n = np.bincount(img_of[cmask], minlength=nimg)
            dsum = np.bincount(img_of[cmask], weights=(hc - hr)[cmask], minlength=nimg)
            # 점추정 = 캡션 단위 차이(표의 두 R@1 의 차와 같음). 범위 = 이미지를 재표집하고 캡션 수로 가중
            ds, ns = dsum[imgs], n[imgs].astype(float)
            idx = rng.integers(0, len(imgs), (1000, len(imgs)))
            lo, hi = np.percentile(ds[idx].sum(axis=1) / ns[idx].sum(axis=1) * 100, [2.5, 97.5])
            rows.append((name, int(cmask.sum()), len(imgs), hr[cmask].mean() * 100, hc[cmask].mean() * 100,
                         (hc[cmask].mean() - hr[cmask].mean()) * 100, lo, hi))
        for r in rows:
            L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:+.2f} [{r[6]:+.2f}, {r[7]:+.2f}] |")
        # 가장 많이 나빠진 사진 15장 (이미지별 hit@1 차이)
        pi_r = R.per_image(np.load(pr)["ranks"], img_of, nimg, 1); pi_c = R.per_image(np.load(pc)["ranks"], img_of, nimg, 1)
        worst = np.argsort(pi_c - pi_r)[:15]
        L += ["", f"가장 많이 나빠진 사진 15장 (이미지별 R@1 FP32 → 양자화): " +
              ", ".join(f"{files[i]} ({pi_r[i]:.1f}→{pi_c[i]:.1f})" for i in worst),
              f"- 캡션 단위 뒤집힘: FP32만 정답 {int(((hr == 1) & (hc == 0)).sum())} / 양자화만 정답 {int(((hr == 0) & (hc == 1)).sum())}", ""]
        out[cand] = rows
    L += ["## 읽는 법", "", "- 여러 묶음을 동시에 보므로 우연히 하나쯤 범위가 0을 벗어날 수 있다. 큰 묶음에서 일관된 방향만 결론으로 쓴다",
          "- 사후 분석이다. 구성 선택이나 판정을 바꾸지 않는다"]
    (C.RESULTS / f"failure_analysis_{a.split}.md").write_text("\n".join(L) + "\n")
    (C.RESULTS / f"failure_analysis_{a.split}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
