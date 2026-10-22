"""블록 제거 + 선형 대체 (ReplaceMe, arXiv 2505.02819). 학습 없음, 최소제곱 닫힌 해 1회.

  python -m src.replaceme --n 1            # 연속 n 개 블록 제거, 자르는 위치는 자동 (코사인 거리 최소)
  python -m src.replaceme --n 2 --start 9  # 위치 지정

방법
- 캘리브: COCO 제외 목록 189장(평가에 쓰지 않는 사진) x 197 토큰 = 약 3.7만 행
- 블록 i..i+n-1 을 지우고, 바로 앞 블록(i-1)의 MLP 출력 m 에 768x768 선형 변환 T 를 곱해 대신한다
    목표: a_{i-1} + m_{i-1} T ≈ h_{i+n-1}   (a = MLP 직전 잔차, h = 블록 출력)
    T = argmin ||m T - (h - a)||² (+ 아주 작은 리지)  ->  c_proj 가중치·편향에 접어 넣음 (W' = Tᵀ W, b' = b T)
  → 그래프 모양은 '블록 수만 줄어든 ViT' 그대로 (Hailo 에도 고정 그래프로 들어감)
- 자르는 위치: 캘리브에서 h_{i-1} 과 h_{i+n-1} 의 평균 코사인 거리가 가장 작은 곳 (논문 방식)
결과: <MODELS>/rm/clip_vitb16_rm_n<n>_s<i>.onnx (+ .json), 콘솔에 후보 위치별 거리와 CIFAR 검증 2000장 zero-shot
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image

from . import common as C
from . import export_onnx as E
from . import coco_retrieval as R

OUT_DIR = C.MODELS / "rm"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--start", type=int, default=0, help="지울 첫 블록 (0 = 자동)")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--cifar-n", type=int, default=2000)
    ap.add_argument("--cls-weight", type=float, default=1.0, help="최소제곱에서 CLS 토큰 행 가중치 (출력은 CLS 만 쓰므로)")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(8)
    model, _ = E.load_model(C.PRETRAINED)
    V = model.visual
    blocks = V.transformer.resblocks
    NB = len(blocks)
    files = R.read_split("exclude")
    x = torch.from_numpy(np.stack([C.preprocess(Image.open(C.COCO_DIR / f)) for f in files]))
    H, M = {}, {}

    def bh(i):
        def f(mod, inp, out):
            o = out if out.shape[1] == 197 else out.transpose(0, 1)
            H.setdefault(i, []).append(o.reshape(-1, o.shape[-1]).double())
        return f

    def mh(i):
        def f(mod, inp, out):
            o = out if out.shape[1] == 197 else out.transpose(0, 1)
            M.setdefault(i, []).append(o.reshape(-1, o.shape[-1]).double())
        return f

    hs = [b.register_forward_hook(bh(i)) for i, b in enumerate(blocks)] + [b.mlp.register_forward_hook(mh(i)) for i, b in enumerate(blocks)]
    with torch.no_grad():
        e0 = []
        for k in range(0, len(x), 16):
            e0.append(model.encode_image(x[k:k + 16]))
    for h in hs:
        h.remove()
    H = {i: torch.cat(v) for i, v in H.items()}; M = {i: torch.cat(v) for i, v in M.items()}
    n = a.n
    dist = {}
    for i in range(1, NB - n + 1):
        cs = torch.nn.functional.cosine_similarity(H[i - 1], H[i + n - 1], dim=-1)
        dist[i] = float((1 - cs).mean())
    print("[rm] 위치별 코사인 거리 (지울 블록 i..i+n-1):", {f"{i}-{i + n - 1}": round(d, 4) for i, d in dist.items()}, flush=True)
    s = a.start or min(dist, key=dist.get)
    p = s - 1
    m, A, tgt = M[p], H[p] - M[p], H[s + n - 1]
    Y = tgt - A
    w = torch.ones(m.shape[0], 1, dtype=m.dtype)
    w[0::197] = a.cls_weight                                  # 행 순서: 이미지별 197 토큰, 0 번이 CLS
    m, Y = m * w.sqrt(), Y * w.sqrt()
    G = m.T @ m
    T = torch.linalg.solve(G + a.ridge * torch.eye(G.shape[0], dtype=G.dtype) * G.diagonal().mean(), m.T @ Y)
    fit = float(1 - ((m @ T - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    cp = blocks[p].mlp.c_proj
    with torch.no_grad():
        Tf = T.float()
        cp.weight.copy_(Tf.T @ cp.weight)
        cp.bias.copy_(cp.bias @ Tf)
    keep = [b for k, b in enumerate(blocks) if not (s <= k < s + n)]
    V.transformer.resblocks = torch.nn.ModuleList(keep)
    if hasattr(V.transformer, "layers"):
        V.transformer.layers = len(keep)
    with torch.no_grad():
        e1 = torch.cat([model.encode_image(x[k:k + 16]) for k in range(0, len(x), 16)])
    e0 = torch.cat(e0); e0 = e0 / e0.norm(dim=-1, keepdim=True); e1 = e1 / e1.norm(dim=-1, keepdim=True)
    cos_cal = float((e0 * e1).sum(-1).mean())
    imgs, labels = C.load_eval_set("val")
    step = max(1, len(imgs) // a.cifar_n); idx = np.arange(0, len(imgs), step)[:a.cifar_n]
    xc = torch.from_numpy(np.stack([C.preprocess(Image.fromarray(imgs[k])) for k in idx]))
    t = torch.from_numpy(np.load(C.TEXT_EMB))
    with torch.no_grad():
        ec = torch.cat([model.encode_image(xc[k:k + 32]) for k in range(0, len(xc), 32)])
    ec = ec / ec.norm(dim=-1, keepdim=True)
    acc = float(((ec @ t.T).argmax(-1).numpy() == labels[idx]).mean())
    print(f"[rm] n{n}_s{s} cls_w={a.cls_weight}: R² {fit:.4f}, 캘리브 코사인 {cos_cal:.4f}, CIFAR {len(idx)}장 {acc:.2%}", flush=True)
    if a.no_export:
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"n{n}_s{s}{a.tag}"
    out = OUT_DIR / f"clip_vitb16_rm_{tag}.onnx"; raw = OUT_DIR / f"clip_vitb16_rm_{tag}.raw.onnx"
    info = {"timestamp": C.now(), "n_removed": n, "removed_blocks": list(range(s, s + n)), "folded_into_block": p,
            "cos_dist_by_start": dist, "lstsq_r2": round(fit, 4), "ridge": a.ridge, "calib": "coco_ret_exclude (189장)",
            "calib_emb_cos_mean": round(cos_cal, 4), "cifar_val_n": int(len(idx)), "cifar_val_top1_torch": acc,
            "blocks_left": len(keep)}
    info.update(E.export(model, raw)); info.update(E.fold_constants(raw, out))
    try:
        raw.unlink()
    except OSError:
        pass
    info["out"] = str(out)
    out.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str))
    print(f"[rm] {tag}: 블록 {s}~{s + n - 1} 제거 → {p} 번 c_proj 에 접음. 최소제곱 R² {fit:.4f}, 캘리브 임베딩 코사인 {cos_cal:.4f}, "
          f"CIFAR 검증 {len(idx)}장 {acc:.2%} (원본 약 70.45%), {info.get('size_mb')} MB -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
