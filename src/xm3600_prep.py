"""XM3600 (Crossmodal-3600) 준비·살펴보기. **모델 평가 없음** (봉인 예정 최종 시험셋)

  python -m src.xm3600_prep download     # 공식 주소에서 captions.zip + images.tgz
  python -m src.xm3600_prep inspect      # 사진 수·영어 캡션 수·길이·77토큰 초과 비율·지역 정보, 예시 사진 8장 docs/samples/xm3600
"""
from __future__ import annotations
import io, json, random, sys, tarfile, urllib.request, zipfile
from pathlib import Path
from . import common as C

DIR = C.DATA / "xm3600"
CAP_URL = "https://google.github.io/crossmodal-3600/web-data/captions.zip"
IMG_URL = "https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz"


def download():
    DIR.mkdir(parents=True, exist_ok=True)
    from .download_data import _download
    z = DIR / "captions.zip"
    if not z.exists(): _download(CAP_URL, z)
    with zipfile.ZipFile(z) as f: f.extractall(DIR)
    t = DIR / "images.tgz"
    if not t.exists(): _download(IMG_URL, t)
    if not (DIR / "images").exists():
        with tarfile.open(t) as f: f.extractall(DIR)
    print("[xm3600] 파일 수:", sum(1 for _ in DIR.iterdir()))


def load_en():
    p = next(DIR.rglob("captions.jsonl"))
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    key = next(k for k in rows[0] if "key" in k)
    out = {r[key]: r["en"]["caption"] for r in rows if "en" in r}
    return rows, key, out


def inspect():
    import numpy as np
    from PIL import Image
    rows, key, en = load_en()
    imgdir = next((p for p in [DIR, *DIR.rglob("*")] if p.is_dir() and any(p.glob("*.jpg"))))
    imgs = {p.stem: p for p in imgdir.glob("*.jpg")}
    n_caps = [len(v) for v in en.values()]
    import open_clip
    tok = open_clip.get_tokenizer(C.MODEL_NAME)
    allc = [c for v in en.values() for c in v]
    lens = [len(c.split()) for c in allc]
    toks = (tok(allc) != 0).sum(1).numpy()
    print(f"[xm3600] 사진 {len(rows)}장 (이미지 파일 {len(imgs)}), 영어 캡션 {len(allc)}개, 사진당 {np.mean(n_caps):.2f} (최소 {min(n_caps)}, 최대 {max(n_caps)})")
    print(f"[xm3600] 영어 캡션 단어 수: 평균 {np.mean(lens):.1f}, 중앙 {np.median(lens):.0f}, 최대 {max(lens)} / 77토큰 초과 {float((toks >= 77).mean()) * 100:.2f}%")
    other = [k for k in rows[0] if k not in (key,) and not isinstance(rows[0][k], dict)]
    print("[xm3600] 영어 외 필드 예:", other[:10], "| 언어 수", sum(isinstance(v, dict) for v in rows[0].values()))
    sizes = [Image.open(p).size for p in list(imgs.values())[:300]]
    print(f"[xm3600] 사진 크기(앞 300장): 긴 변 중앙 {int(np.median([max(s) for s in sizes]))}, 최소 {min(max(s) for s in sizes)}, 최대 {max(max(s) for s in sizes)}")
    out = C.ROOT / "docs" / "samples" / "xm3600"; out.mkdir(parents=True, exist_ok=True)
    random.seed(0); pick = random.sample(sorted(en), 8); L = []
    for k in pick:
        im = Image.open(imgs[k]); im.save(out / f"{k}.jpg", quality=92)
        L.append(f"- {k}.jpg {im.size}: " + " / ".join(en[k]))
    (out / "README.md").write_text("# XM3600 예시 (무작위 8장, 시드 0, 모델 평가 없음)\n\n" + "\n".join(L) + "\n"); print("\n".join(L))
    # COCO 와 문장 스타일 비교용: COCO val 캡션 단어 수
    cap = json.loads((C.DATA / "coco/annotations/captions_val2017.json").read_text())
    cl = [len(a["caption"].split()) for a in cap["annotations"]]
    print(f"[비교] COCO val 캡션 단어 수 평균 {np.mean(cl):.1f}, 중앙 {np.median(cl):.0f}")


if __name__ == "__main__":
    {"download": download, "inspect": inspect}[sys.argv[1]]()
