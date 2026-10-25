"""E43 최종 시험셋 XM3600 (Crossmodal-3600) 영어 캡션 텍스트→이미지 검색. **봉인: 최종 동결 후 한 번만** (eval/table 은 --i-know-final)

  python -m src.xm3600_eval text [--tag T --text-model PATH]
  python -m src.xm3600_eval eval --i-know-final CODE=ONNX[:TEXTTAG] ...
  python -m src.xm3600_eval table --i-know-final --ref A0 --codes ...

- 주 지표: NoCaps 와 겹치는 사진(Open Images id 기준)을 갤러리·질의에서 모두 뺀 3,557장. 보조: 전체 3,600장
- 평가 방식은 COCO 와 같음 (C.preprocess, 캡션 원문, 이미지 평균 R@K, 이미지 단위 짝지은 부트스트랩 2000회)
"""
from __future__ import annotations
import json, sys
import numpy as np
from PIL import Image
from . import common as C
from .coco_retrieval import per_image, ranks
from .xm3600_prep import DIR, load_en

OUT = C.RESULTS / "xm3600"


def load():
    _, _, en = load_en()
    imgdir = next((p for p in [DIR, *DIR.rglob("*")] if p.is_dir() and any(p.glob("*.jpg"))))
    files = sorted(k for k in en if (imgdir / f"{k}.jpg").exists())
    assert len(files) == 3600, f"사진 {len(files)}장"
    img_of = np.array([i for i, k in enumerate(files) for _ in en[k]], dtype=np.int64)
    caps = [c for k in files for c in en[k]]
    noc = json.loads((C.DATA / "nocaps" / "captions.json").read_text())["meta"]
    oid = {str(v.get("open_images_id")) for v in noc.values() if v.get("open_images_id")}
    keep = np.array([k not in oid for k in files])
    return files, img_of, caps, imgdir, keep


def text_file(tag=""):
    return C.STORE / "models" / f"xm3600_text{('_' + tag) if tag else ''}.npy"


def cmd_text(args):
    import open_clip, torch
    tag = args[args.index("--tag") + 1] if "--tag" in args else ""
    p = text_file(tag)
    if p.exists():
        print("[xm3600] 이미 있음", p.name); return
    _, _, caps, _, _ = load()
    m = open_clip.create_model(C.MODEL_NAME, pretrained=C.PRETRAINED, force_quick_gelu=True, cache_dir=str(C.CACHE)).eval()
    if "--text-model" in args:
        m.load_state_dict(torch.load(args[args.index("--text-model") + 1]), strict=False)
    tok = open_clip.get_tokenizer(C.MODEL_NAME); out = []
    with torch.no_grad():
        for i in range(0, len(caps), 256):
            e = m.encode_text(tok(caps[i:i + 256])).float().numpy(); out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    np.save(p, np.concatenate(out)); print(f"[xm3600] 캡션 임베딩 {len(caps)}개 → {p.name}")


def _sub(emb, txt, img_of, keep):
    new = -np.ones(len(keep), dtype=np.int64); new[keep] = np.arange(keep.sum())
    cm = keep[img_of]
    return ranks(emb[keep], txt[cm], new[img_of[cm]]), new[img_of[cm]]


def cmd_eval(pairs):
    from .backends import OnnxBackend
    files, img_of, _, imgdir, keep = load(); OUT.mkdir(parents=True, exist_ok=True)
    print(f"[xm3600] 사진 {len(files)}장, 캡션 {len(img_of)}개, NoCaps 겹침 {int((~keep).sum())}장 → 주 지표 {int(keep.sum())}장", flush=True)
    for pr in pairs:
        code, rest = pr.split("=", 1); path, _, tag = rest.partition(":")
        if (OUT / f"{code}.npz").exists(): continue
        txt = np.load(text_file(tag)); be = OnnxBackend(path, code)
        emb = np.stack([be.encode_image(C.preprocess(Image.open(imgdir / f"{f}.jpg"))[None]) for f in files]).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        r_all = ranks(emb, txt, img_of); r_main, io_main = _sub(emb, txt, img_of, keep)
        np.savez_compressed(OUT / f"{code}.npz", r_all=r_all, img_of=img_of, r_main=r_main, io_main=io_main)
        print(f"[xm3600] {code}: R@1 주(3,557) {per_image(r_main, io_main, int(keep.sum()), 1).mean():.2%} / 전체 {per_image(r_all, img_of, len(files), 1).mean():.2%}", flush=True)


def _block(Dd, codes, ref, rk, ik, n, title):
    idx = np.random.default_rng(0).integers(0, n, size=(2000, n))
    base = per_image(Dd[ref][rk], Dd[ref][ik], n, 1)
    L = [f"## {title} (갤러리 {n}장 / 캡션 {len(Dd[ref][ik])}개)", "",
         f"| 구성 | R@1 | R@5 | R@10 | R@1 차이 vs {ref} [95% CI] |", "|---|---|---|---|---|"]
    for c in codes:
        pk = {k: per_image(Dd[c][rk], Dd[c][ik], n, k) for k in (1, 5, 10)}
        d = pk[1] - base; lo, hi = np.percentile(d[idx].mean(1) * 100, [2.5, 97.5])
        L.append(f"| {c} | {pk[1].mean():.2%} | {pk[5].mean():.2%} | {pk[10].mean():.2%} | {d.mean() * 100:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    return L


def cmd_table(ref, codes, pair):
    files, _, _, _, keep = load(); Dd = {c: np.load(OUT / f"{c}.npz") for c in codes}
    L = ["# XM3600 영어 텍스트→이미지 (E43 최종 시험셋, 한 번만 연다)", "",
         "이미지 평균 R@K, 95% CI = 이미지 단위 짝지은 부트스트랩 2000회", ""]
    L += _block(Dd, codes, ref, "r_main", "io_main", int(keep.sum()), "주 지표: NoCaps 겹침 43장 제외")
    L += [""] + _block(Dd, codes, ref, "r_all", "img_of", len(files), "보조: 전체 3,600장")
    if pair:
        a, b = pair
        L += ["", f"## 규칙 4 확인: {a} vs {b} (주 지표)", ""] + _block(Dd, [b, a], b, "r_main", "io_main", int(keep.sum()), f"{a} vs {b}")[2:]
    (C.RESULTS / "xm3600.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "text": cmd_text(a)
    elif "--i-know-final" not in a: sys.exit("XM3600 은 최종 동결 후 한 번만 엽니다 (E43). --i-know-final 필요")
    elif a[0] == "eval": cmd_eval([x for x in a[1:] if "=" in x])
    elif a[0] == "table":
        pair = a[a.index("--pair") + 1].split(",") if "--pair" in a else None
        stop = [i for i, x in enumerate(a) if x.startswith("--") and i > a.index("--codes")]
        codes = a[a.index("--codes") + 1:(stop[0] if stop else None)]
        cmd_table(a[a.index("--ref") + 1], codes, pair)
