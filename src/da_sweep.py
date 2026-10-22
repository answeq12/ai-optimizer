"""E33: 섞는 비율 α 탐색과 판단 ① (사전 등록 규칙 그대로).

  python -m src.da_sweep --ckpt da_tail_ep2.pt

COCO-select R@1(이미지 평균) 과 CIFAR val top-1 을 저장 특징(블록 7 출력, fp16) 위에서 뒤쪽만 돌려 잰다.
규칙: CIFAR val 이 α=0 대비 −1.0%p 이내인 α 중 select R@1 최고. 판단 ①: 그 값 − α=0 ≥ +2.0%p.
통과하면 models_da/da_tail_chosen.pt 저장, 종료코드 0. 실패하면 종료코드 3.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import common as C
from . import da_common as D
from .coco_retrieval import captions_for, per_image, ranks

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
CIFAR_TOL, GATE1 = 1.0, 2.0


def embed(tail, feat, bs=64):
    import torch
    out = np.empty((len(feat), 512), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(feat), bs):
            e = tail(torch.from_numpy(np.asarray(feat[s:s + bs], dtype=np.float32))).numpy()
            out[s:s + len(e)] = e / np.linalg.norm(e, axis=1, keepdims=True)
    return out


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="da_tail_ep2.pt")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tag", default="", help="결과·선택 가중치 이름 꼬리표 (비우면 E33 이름 = 최종 가중치를 덮어씀)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    th0 = torch.load(D.DA_DIR / "da_tail_init.pt")
    th1 = torch.load(D.DA_DIR / a.ckpt)
    m = D.load_clip()
    tail = D.make_tail(m.visual).eval()
    fs, _ = D.open_feat("select"); fc, _ = D.open_feat("cifarval")
    _, img_of, _ = captions_for("select")
    txt = np.load(C.MODELS / "coco_ret_text_select.npy")
    temb = np.load(C.TEXT_EMB)
    _, lab = C.load_eval_set("val")
    nimg = int(img_of.max() + 1)
    rows = []
    for al in ALPHAS:
        tail.load_state_dict({k: (1 - al) * th0[k] + al * th1[k] for k in th0})
        es, ec = embed(tail, fs), embed(tail, fc)
        r = ranks(es, txt, img_of)
        row = {"alpha": al, **{f"R@{k}": float(per_image(r, img_of, nimg, k).mean() * 100) for k in (1, 5, 10)},
               "cifar_val": float((np.argmax(ec @ temb.T, 1) == lab).mean() * 100)}
        rows.append(row)
        print(f"[da-sweep] α={al}: select R@1 {row['R@1']:.2f} R@5 {row['R@5']:.2f} R@10 {row['R@10']:.2f} / CIFAR val {row['cifar_val']:.2f}", flush=True)
    base = rows[0]
    ok = [r for r in rows[1:] if r["cifar_val"] >= base["cifar_val"] - CIFAR_TOL]
    best = max(ok, key=lambda r: r["R@1"]) if ok else None
    gain = best["R@1"] - base["R@1"] if best else float("nan")
    passed = bool(best and gain >= GATE1)
    ref = {}
    for k, p in (("A0_select_R@1_table", None), ("A0_cifar_val", C.RESULTS / "preds" / "A0-val.json")):
        if p and p.exists():
            ref[k] = json.loads(p.read_text())["top1_acc"] * 100
    res = {"ckpt": a.ckpt, "rows": rows, "chosen_alpha": best["alpha"] if best else None, "gain_R@1": gain,
           "gate1_threshold": GATE1, "cifar_tol": CIFAR_TOL, "gate1_pass": passed, "reference": ref}
    (C.RESULTS / f"da_sweep{('_' + a.tag) if a.tag else ''}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    L = ["# E33 도메인 적응: 섞는 비율 α 탐색 (COCO-select, CIFAR val)", "",
         "저장 특징(블록 7 출력 fp16) 위에서 뒤쪽만 계산. α=0 = 원본 모델. R@K 는 이미지 평균(비교표와 같은 방식).", "",
         "| α | select R@1 | R@5 | R@10 | CIFAR val | 규칙 |", "|---|---|---|---|---|---|"]
    for r in rows:
        tag = "원본" if r["alpha"] == 0 else ("**선택**" if best and r is best else
                                             ("CIFAR 제약 위반" if r not in ok else ""))
        L.append(f"| {r['alpha']} | {r['R@1']:.2f} | {r['R@5']:.2f} | {r['R@10']:.2f} | {r['cifar_val']:.2f} | {tag} |")
    L += ["", f"- 판단 ①: 선택 α 의 R@1 − α=0 = {gain:+.2f}%p (기준 +{GATE1}) → **{'통과' if passed else '미통과'}**",
          f"- 참고: 원본 ONNX 의 CIFAR val {ref.get('A0_cifar_val', float('nan')):.2f}% (α=0 과 같아야 함, fp16 저장 영향만큼 차이 가능)",
          "- 사전 등록: docs/02 E33"]
    (C.RESULTS / f"da_sweep{('_' + a.tag) if a.tag else ''}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    if passed:
        torch.save({k: (1 - best["alpha"]) * th0[k] + best["alpha"] * th1[k] for k in th0}, D.DA_DIR / f"da_tail_chosen{('_' + a.tag) if a.tag else ''}.pt")
        (D.DA_DIR / f"da_tail_chosen{('_' + a.tag) if a.tag else ''}.json").write_text(json.dumps({"alpha": best["alpha"], "ckpt": a.ckpt}))
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()
