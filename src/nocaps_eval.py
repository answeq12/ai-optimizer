"""E34 최종 테스트셋 NoCaps val 텍스트→이미지 검색. **봉인: 최종 동결 후 한 번만** (eval/table 은 --i-know-final 필요)

  python -m src.nocaps_eval text [--tag T --text-model PATH]   # 캡션 임베딩 (원본 FP32 텍스트 인코더 = 기본)
  python -m src.nocaps_eval eval --i-know-final CODE=ONNX[:TEXTTAG] ...
  python -m src.nocaps_eval table --i-know-final --ref A0 --codes ...
평가 방식은 COCO 와 같음 (C.preprocess, 캡션 원문, 이미지 평균 R@K, 이미지 단위 짝지은 부트스트랩)
"""
from __future__ import annotations
import json, sys
import numpy as np
from PIL import Image
from . import common as C
from .coco_retrieval import per_image, ranks

DIR = C.DATA / "nocaps"
OUT = C.RESULTS / "nocaps"


def load():
    d = json.loads((DIR / "captions.json").read_text())
    files = sorted(d["captions"])
    img_of = np.array([k for k, f in enumerate(files) for _ in d["captions"][f]], dtype=np.int64)
    return files, img_of, [c for f in files for c in d["captions"][f]], d["meta"]


def text_file(tag=""):
    return C.STORE / "models" / f"nocaps_text{('_' + tag) if tag else ''}.npy"


def cmd_text(args):
    import open_clip, torch
    tag = args[args.index("--tag") + 1] if "--tag" in args else ""
    p = text_file(tag)
    if p.exists():
        print("[nocaps] 이미 있음", p); return
    _, _, caps, _ = load()
    m = open_clip.create_model(C.MODEL_NAME, pretrained=C.PRETRAINED, force_quick_gelu=True, cache_dir=str(C.CACHE)).eval()
    if "--text-model" in args:   # 텍스트 쪽을 학습한 경우: 상태 사전 덮어쓰기
        m.load_state_dict(torch.load(args[args.index("--text-model") + 1]), strict=False)
    tok = open_clip.get_tokenizer(C.MODEL_NAME); out = []
    with torch.no_grad():
        for i in range(0, len(caps), 256):
            e = m.encode_text(tok(caps[i:i + 256])).float().numpy(); out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    np.save(p, np.concatenate(out)); print(f"[nocaps] 캡션 임베딩 {len(caps)}개 → {p.name}")


def cmd_eval(pairs):
    from .backends import OnnxBackend
    files, img_of, _, _ = load(); OUT.mkdir(parents=True, exist_ok=True)
    for pr in pairs:
        code, rest = pr.split("=", 1); path, _, tag = rest.partition(":")
        if (OUT / f"{code}.npz").exists(): continue
        txt = np.load(text_file(tag)); be = OnnxBackend(path, code)
        emb = np.stack([be.encode_image(C.preprocess(Image.open(DIR / "images" / f))[None]) for f in files]).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        r = ranks(emb, txt, img_of); np.savez_compressed(OUT / f"{code}.npz", ranks=r, img_of=img_of)
        print(f"[nocaps] {code}: R@1 {per_image(r, img_of, len(files), 1).mean():.2%}", flush=True)


def cmd_table(ref, codes):
    files, img_of, _, meta = load(); n = len(files)
    D = {c: np.load(OUT / f"{c}.npz") for c in codes}
    idx = np.random.default_rng(0).integers(0, n, size=(2000, n))
    dom = np.array([meta[f].get("domain") or "?" for f in files])
    L = [f"# NoCaps val 텍스트→이미지 (최종 테스트셋 E34, 갤러리 {n}장 / 캡션 {len(img_of)}개)", "",
         "이미지 평균 R@K, 95% CI = 이미지 단위 짝지은 부트스트랩 2000회", "",
         f"| 구성 | R@1 | R@5 | R@10 | R@1 차이 vs {ref} [95% CI] |", "|---|---|---|---|---|"]
    base = per_image(D[ref]["ranks"], img_of, n, 1)
    for c in codes:
        pk = {k: per_image(D[c]["ranks"], img_of, n, k) for k in (1, 5, 10)}
        d = pk[1] - base; lo, hi = np.percentile(d[idx].mean(1) * 100, [2.5, 97.5])
        L.append(f"| {c} | {pk[1].mean():.2%} | {pk[5].mean():.2%} | {pk[10].mean():.2%} | {d.mean()*100:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    if (dom != "?").any():
        L += ["", "## 묶음별 R@1 (갤러리는 전체 4,500장 그대로, 해당 묶음 사진의 성적만 평균)", "",
              "| 구성 | " + " | ".join(sorted(set(dom))) + " |", "|---|" + "---|" * len(set(dom))]
        for c in codes:
            p1 = per_image(D[c]["ranks"], img_of, n, 1)
            L.append(f"| {c} | " + " | ".join(f"{p1[dom == g].mean():.2%} (n={int((dom == g).sum())})" for g in sorted(set(dom))) + " |")
    (C.RESULTS / "nocaps.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "text": cmd_text(a)
    elif "--i-know-final" not in a: sys.exit("NoCaps 는 최종 동결 후 한 번만 엽니다 (E34). --i-know-final 필요")
    elif a[0] == "eval": cmd_eval([x for x in a[1:] if "=" in x])
    elif a[0] == "table": cmd_table(a[a.index("--ref") + 1], [x for x in a[a.index("--codes") + 1:] if not x.startswith("--")])
