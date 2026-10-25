"""E46: CC3M (Conceptual Captions) — COCO 밖 사진 추가 학습 데이터 + COCO 밖 선택용 검증셋.

  python -m src.cc3m_eval prepare --train-shards 20 --val-shards 3   # HF pixparse/cc3m-wds 샤드 받기·풀기
  python -m src.cc3m_eval text --tag da3 --text-model /mnt/d/halio_data/models_da3/text_state.pt
  python -m src.cc3m_eval eval A0=path.onnx DA3-SPL=path.onnx:da3
  python -m src.cc3m_eval table --ref DA3-SPL --codes A0 DA3-SPL DA5-SPL

- 학습: train 샤드 앞쪽에서 100,000장 (키 정렬 순). 사진당 캡션 1개(웹 alt-text 를 정리한 문장)
- 검증(선택용, 시험셋 아님): validation 샤드 앞쪽에서 2,000장. 텍스트→이미지 R@1 (갤러리 2,000장, 캡션=사진 1:1)
- 사진은 img2dataset 로 짧은 변 512 로 줄여 저장된 것. 평가·학습 전처리는 COCO 와 같다(C.preprocess)
"""
from __future__ import annotations
import io, json, sys, tarfile
from pathlib import Path
import numpy as np
from PIL import Image
from . import common as C
from .coco_retrieval import per_image, ranks

REPO = "pixparse/cc3m-wds"
DIR = C.DATA / "cc3m"
OUT = C.RESULTS / "cc3m"
N_TRAIN, N_VAL = 100000, 2000
TRAIN_LIST = C.SPLITS / "cc3m_train_100k.txt"
VAL_LIST = C.SPLITS / "cc3m_val_2k.txt"


def _extract(tar_path, dst: Path, caps: dict):
    """jpg + txt 만, 이름은 키의 basename 만 쓴다(경로 조작 방지). 깨진 사진은 건너뜀."""
    recs = {}
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            if not m.isfile() or "." not in m.name:
                continue
            key, ext = Path(m.name).name.rsplit(".", 1)
            if ext in ("jpg", "txt"):
                recs.setdefault(key, {})[ext] = t.extractfile(m).read()
    ok = 0
    for key, r in sorted(recs.items()):
        if "jpg" not in r or "txt" not in r:
            continue
        cap = r["txt"].decode("utf-8", "ignore").strip()
        if not cap:
            continue
        try:
            im = Image.open(io.BytesIO(r["jpg"])); im.verify()
        except Exception:
            continue
        (dst / f"{key}.jpg").write_bytes(r["jpg"]); caps[f"{key}.jpg"] = cap; ok += 1
    return ok


def cmd_prepare(args):
    from huggingface_hub import hf_hub_download
    nt = int(args[args.index("--train-shards") + 1]) if "--train-shards" in args else 20
    nv = int(args[args.index("--val-shards") + 1]) if "--val-shards" in args else 3
    sh = DIR / "_shards"; sh.mkdir(parents=True, exist_ok=True)
    for split, n, lst, need, sub in (("validation", nv, VAL_LIST, N_VAL, "val"), ("train", nt, TRAIN_LIST, N_TRAIN, "train")):
        if lst.exists():
            print(f"[cc3m] {lst.name} 이미 있음"); continue
        dst = DIR / sub; dst.mkdir(parents=True, exist_ok=True); caps = {}
        for i in range(n):
            f = hf_hub_download(REPO, f"cc3m-{split}-{i:04d}.tar", repo_type="dataset", local_dir=str(sh))
            got = _extract(f, dst, caps); print(f"[cc3m] {split} 샤드 {i}: {got}장 (누적 {len(caps)})", flush=True)
            if len(caps) >= need:
                break
        if len(caps) < need:
            sys.exit(f"[cc3m] {split} {len(caps)}장 < {need} → 샤드 수를 늘릴 것")
        names = sorted(caps)[:need]
        (DIR / f"captions_{sub}.json").write_text(json.dumps({n_: caps[n_] for n_ in names}, ensure_ascii=False))
        lst.write_text(f"# source=pixparse/cc3m-wds {split} 앞 {n}샤드, 키 정렬 앞 {need}장\n" + "\n".join(names) + "\n")
        print(f"[cc3m] {split}: {need}장 확정 → {lst.name}", flush=True)
    # COCO 와 겹침 없음은 출처가 달라 자명(웹 수집). 검증·학습 겹침 확인
    tr = set(C.SPLITS.joinpath(TRAIN_LIST.name).read_text().split("\n")[1:]); va = set(VAL_LIST.read_text().split("\n")[1:])
    print(f"[cc3m] train/val 파일명 겹침 {len((tr & va) - {''})}개")


def captions(sub):
    return json.loads((DIR / f"captions_{sub}.json").read_text())


def load_val():
    d = captions("val"); files = [ln for ln in VAL_LIST.read_text().split("\n")[1:] if ln]
    return files, np.arange(len(files)), [d[f] for f in files]


def text_file(tag=""):
    return C.STORE / "models" / f"cc3m_val_text{('_' + tag) if tag else ''}.npy"


def cmd_text(args):
    import open_clip, torch
    tag = args[args.index("--tag") + 1] if "--tag" in args else ""
    p = text_file(tag)
    if p.exists():
        return
    _, _, caps = load_val()
    m = open_clip.create_model(C.MODEL_NAME, pretrained=C.PRETRAINED, force_quick_gelu=True, cache_dir=str(C.CACHE)).eval()
    if "--text-model" in args:
        m.load_state_dict(torch.load(args[args.index("--text-model") + 1]), strict=False)
    tok = open_clip.get_tokenizer(C.MODEL_NAME); out = []
    with torch.no_grad():
        for i in range(0, len(caps), 256):
            e = m.encode_text(tok(caps[i:i + 256])).float().numpy(); out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    np.save(p, np.concatenate(out)); print(f"[cc3m] 검증 캡션 임베딩 {len(caps)}개 ({tag or '원본'})")


def cmd_eval(pairs):
    from .backends import OnnxBackend
    files, img_of, _ = load_val(); OUT.mkdir(parents=True, exist_ok=True)
    for pr in pairs:
        code, rest = pr.split("=", 1); path, _, tag = rest.partition(":")
        if (OUT / f"{code}.npz").exists():
            continue
        txt = np.load(text_file(tag)); be = OnnxBackend(path, code)
        emb = np.stack([be.encode_image(C.preprocess(Image.open(DIR / "val" / f))[None]) for f in files])
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        r = ranks(emb.astype(np.float32), txt, img_of)
        np.savez_compressed(OUT / f"{code}.npz", ranks=r, img_of=img_of)
        print(f"[cc3m] {code}: R@1 {per_image(r, img_of, len(files), 1).mean():.2%}", flush=True)


def cmd_table(ref, codes, tag="cc3m_val"):
    Dd = {c: np.load(OUT / f"{c}.npz") for c in codes}
    img_of = Dd[ref]["img_of"]; n = int(img_of.max() + 1)
    idx = np.random.default_rng(0).integers(0, n, size=(2000, n))
    base = per_image(Dd[ref]["ranks"], img_of, n, 1)
    L = [f"# CC3M validation 2K 텍스트→이미지 (COCO 밖 선택용 검증셋, 갤러리 {n}장 / 캡션 {len(img_of)}개)", "",
         "이미지 평균 R@K, 95% CI = 이미지 단위 짝지은 부트스트랩 2000회. 시험셋 아님(E46 선택용)", "",
         f"| 구성 | R@1 | R@5 | R@10 | R@1 차이 vs {ref} [95% CI] |", "|---|---|---|---|---|"]
    for c in codes:
        pk = {k: per_image(Dd[c]["ranks"], img_of, n, k) for k in (1, 5, 10)}
        d = pk[1] - base; lo, hi = np.percentile(d[idx].mean(1) * 100, [2.5, 97.5])
        L.append(f"| {c} | {pk[1].mean():.2%} | {pk[5].mean():.2%} | {pk[10].mean():.2%} | {d.mean() * 100:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    (C.RESULTS / f"{tag}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare": cmd_prepare(sys.argv[2:])
    elif cmd == "text": cmd_text(sys.argv[2:])
    elif cmd == "eval": cmd_eval(sys.argv[2:])
    elif cmd == "table":
        a = sys.argv[2:]; ref = a[a.index("--ref") + 1]; codes = a[a.index("--codes") + 1:]
        cmd_table(ref, codes)
