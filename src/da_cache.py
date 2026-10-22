"""E33: 블록 7 출력(얼린 앞부분) 저장 + 학습 캡션 텍스트 임베딩.

  python -m src.da_cache check                    # 앞/뒤로 나눈 계산 = 원래 모델 출력 확인 (필수)
  python -m src.da_cache images --set train --n 2000
  python -m src.da_cache images --set select      # COCO-select 2,405장 (α 탐색용)
  python -m src.da_cache images --set cifarval    # CIFAR-100 val 10,000장 (α 제약용)
  python -m src.da_cache text --n 30000           # 학습 사진 캡션 임베딩 (FP32 텍스트 인코더)
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image

from . import common as C
from . import da_common as D


def _prep_file(path):
    return C.preprocess(Image.open(path))


def _prep_arr(a):
    return C.preprocess(Image.fromarray(a))


def maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cmd_check():
    import torch
    m = D.load_clip()
    v = m.visual
    tail = D.make_tail(v)
    files = D.read_list(C.SPLITS / "calib_coco_64_s0.txt")[:8]
    x = torch.from_numpy(np.stack([C.preprocess(Image.open(C.COCO_DIR / f)) for f in files]))
    with torch.no_grad():
        a = v(x).numpy()
        f = D.front(v, x)
        b = tail(f).numpy()
        b16 = tail(f.half().float()).numpy()
    err = float(np.abs(a - b).max())
    cos16 = float(np.min(np.sum(a * b16, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b16, axis=1)))
    ok = err <= 1e-4
    D.FEAT_DIR.mkdir(parents=True, exist_ok=True)
    (C.LOGS / "da_split_check.json").write_text(json.dumps(
        {"max_abs_err_split_vs_full": err, "min_cos_fp16_cache": cos16, "block7_absmax": float(f.abs().max()),
         "pass": ok}, indent=2))
    print(f"[da] 앞/뒤 나눔 = 원래 출력: 최대 오차 {err:.2e} -> {'통과' if ok else '실패'}; "
          f"fp16 저장 시 최소 코사인 {cos16:.6f}; 블록7 출력 최대 |x| {float(f.abs().max()):.1f}")
    if not ok:
        sys.exit(1)


def cmd_images(a):
    import torch
    torch.set_num_threads(a.threads)
    if a.set == "train":
        files = D.read_list(D.train_list(a.n))[:a.n]
        items, fn = [str(D.TRAIN_DIR / f) for f in files], _prep_file
    elif a.set == "select":
        files = D.read_list(C.SPLITS / "coco_ret_select.txt")
        items, fn = [str(C.COCO_DIR / f) for f in files], _prep_file
    elif a.set == "cifarval":
        imgs, _ = C.load_eval_set("val")
        items, fn = list(imgs), _prep_arr
    name = f"{a.set}" + (f"_{a.n}" if a.set == "train" else "")
    n = len(items)
    D.FEAT_DIR.mkdir(parents=True, exist_ok=True)
    mp = D.feat_path(name)
    if mp.exists() and (D.FEAT_DIR / f"{name}.json").exists():
        print(f"[da] {name} 이미 있음 ({n}장)")
        return
    out = np.memmap(str(mp) + ".part", dtype=np.float16, mode="w+", shape=(n, 197, 768))
    v = D.load_clip().visual
    t0 = time.time()
    bs = a.batch
    s_start = 0
    if a.set == "train" and n > 30000 and (D.FEAT_DIR / "train_30000.json").exists():
        # E37: 앞 30,000장은 E33 저장분과 같은 사진·같은 코드 → 복사 후 8장 다시 계산해 확인
        old, _ = D.open_feat("train_30000")
        for s in range(0, 30000, 4096):
            out[s:min(30000, s + 4096)] = old[s:min(30000, s + 4096)]
        chk = np.random.default_rng(0).choice(30000, 8, replace=False)
        with torch.no_grad():
            re = D.front(v, torch.from_numpy(np.stack([fn(items[i]) for i in chk]))).numpy()
        err = float(np.abs(re.astype(np.float16).astype(np.float32) - np.asarray(old[chk], dtype=np.float32)).max())
        print(f"[da] 앞 30,000장 재사용 (다시 계산 8장 최대 차 {err:.2e})", flush=True)
        assert err < 0.5, "재사용 확인 실패"
        s_start = 30000
    # E37: imap 은 앞서 읽기에 한도가 없어 118k장에서 메모리 부족(OOM)으로 죽었다 → 1,024장 단위로 나눠 처리
    blk = 1024
    with Pool(a.workers) as pool, torch.no_grad():
        for b0 in range(s_start, n, blk):
            pre = pool.map(fn, items[b0:min(n, b0 + blk)], chunksize=8)
            for s in range(b0, min(n, b0 + blk), bs):
                xb = np.stack(pre[s - b0:s - b0 + bs])
                out[s:s + len(xb)] = D.front(v, torch.from_numpy(xb)).numpy().astype(np.float16)
            del pre
            if (b0 // blk) % 16 == 0:
                out.flush()
            dt = time.time() - t0; done = min(n, b0 + blk)
            print(f"[da] {name} {done}/{n} ({dt:.0f}s, 이번 실행 {(done - s_start) / dt:.1f}장/s, 최대 메모리 {maxrss_mb():.0f} MB)", flush=True)
    out.flush(); del out
    import os
    os.replace(str(mp) + ".part", mp)
    dt = time.time() - t0
    D.write_meta(name, n, {"set": a.set, "seconds": round(dt, 1), "per_image_ms": round(dt / n * 1000, 2)})
    print(f"[da] {name}: {n}장 {dt:.0f}s ({dt / n * 1000:.1f} ms/장), {mp.stat().st_size / 1e9:.2f} GB, 최대 메모리 {maxrss_mb():.0f} MB", flush=True)


def cmd_text(a):
    import open_clip
    import torch
    torch.set_num_threads(a.threads)
    from .da_data import captions
    ann = captions()
    files = D.read_list(D.train_list(a.n))[:a.n]
    idx = {fn: k for k, fn in enumerate(files)}
    fid = {im["id"]: im["file_name"] for im in ann["images"]}
    caps = sorted((idx[fid[c["image_id"]]], c["id"], c["caption"].strip()) for c in ann["annotations"]
                  if fid[c["image_id"]] in idx)
    out_e, out_i = D.FEAT_DIR / f"train_{a.n}_text.npy", D.FEAT_DIR / f"train_{a.n}_text_img.npy"
    if out_e.exists():
        print(f"[da] {out_e.name} 이미 있음")
        return
    m = open_clip.create_model(C.MODEL_NAME, pretrained=C.PRETRAINED, force_quick_gelu=True,
                               cache_dir=str(C.CACHE)).eval()
    tok = open_clip.get_tokenizer(C.MODEL_NAME)
    txt = [c[2] for c in caps]
    E = np.empty((len(txt), 512), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(txt), 256):
            e = m.encode_text(tok(txt[i:i + 256])).float().numpy()
            E[i:i + len(e)] = e / np.linalg.norm(e, axis=1, keepdims=True)
            if (i // 256) % 40 == 0:
                print(f"[da] 캡션 {i + len(e)}/{len(txt)} ({time.time() - t0:.0f}s)", flush=True)
    np.save(out_e, E)
    np.save(out_i, np.array([c[0] for c in caps], dtype=np.int64))
    print(f"[da] 캡션 {len(txt)}개 (사진 {len(files)}장) {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    pi = sub.add_parser("images")
    pi.add_argument("--set", required=True, choices=["train", "select", "cifarval"])
    pi.add_argument("--n", type=int, default=30000)
    pi.add_argument("--batch", type=int, default=32)
    pi.add_argument("--threads", type=int, default=12)
    pi.add_argument("--workers", type=int, default=4)
    pt = sub.add_parser("text")
    pt.add_argument("--n", type=int, default=30000)
    pt.add_argument("--threads", type=int, default=14)
    a = ap.parse_args()
    {"check": cmd_check, "images": lambda: cmd_images(a), "text": lambda: cmd_text(a)}[a.cmd]()


if __name__ == "__main__":
    main()
