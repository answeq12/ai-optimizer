"""레지스터 뉴런 진단 (E23, 10/17). 학습 없음, 정확도·통계만.

질문 1. 우리가 찾은 fc2 입력 이상치 채널(블록 5~7 MLP 뉴런)이 '레지스터 뉴런'(Jiang 외, NeurIPS 2025, arXiv 2506.08010)과 같은가?
  - 레지스터 뉴런: 잔차 스트림 노름이 비정상적으로 큰 '이상치 토큰'에서 강하게 켜지는 소수 MLP 뉴런
질문 2. 테스트 시 레지스터(토큰 1개 추가 + 그 뉴런 값을 추가 토큰으로 옮김)를 하면
  - 임베딩/정확도는 유지되는가 (원본 대비 코사인, CIFAR 검증 앞 2000장 zero-shot)
  - 이미지 토큰(추가 토큰 제외)의 fc2 입력 범위(absmax, 99.9% 눈금 칸 수)가 줄어드는가 -> 8비트만으로 버틸 가능성
데이터: COCO 제외 목록(coco_ret_exclude, 캘리브레이션 합집합 189장)만 사용. select/confirm 은 쓰지 않는다
  python -m src.register_probe [--top-k 10] [--top-layer 7]
결과: results/register_probe.json, results/register_probe.md
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from PIL import Image

from . import common as C
from . import coco_retrieval as R


def bn(h, nt):
    """[B,N,D] 또는 [N,B,D] -> [B,N,D]"""
    return h if h.shape[1] == nt else h.transpose(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--top-layer", type=int, default=7, help="이 블록까지(포함)에서 레지스터 뉴런을 찾는다")
    ap.add_argument("--cifar-n", type=int, default=2000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--n-images", type=int, default=96, help="메모리 때문에 제외 목록 앞 N장")
    a = ap.parse_args()
    import open_clip
    torch.set_num_threads(8)
    model = open_clip.create_model(C.MODEL_NAME, pretrained=C.PRETRAINED, force_quick_gelu=True, cache_dir=str(C.CACHE)).eval()
    V = model.visual
    blocks = V.transformer.resblocks
    files = R.read_split("exclude")[:a.n_images]
    x = torch.from_numpy(np.stack([C.preprocess(Image.open(C.COCO_DIR / f)) for f in files]))
    print(f"[reg] 이미지 {len(x)}장 (coco_ret_exclude)", flush=True)

    state = {"mode": "record", "neurons": {}, "nt": 197, "store": True}
    rec = {"resid": {}, "hid": {}}

    def hid_hook(i):
        def f(mod, inp, out):
            h = bn(out, state["nt"])
            if state["mode"] == "register" and i in state["neurons"]:
                h = h.clone()
                for n in state["neurons"][i]:
                    v = h[:, 1:-1, n].max(dim=1).values          # 이미지 패치 토큰 중 최댓값 (CLS·레지스터 제외)
                    h[:, :, n] = 0.0
                    h[:, -1, n] = v                              # 마지막 = 레지스터 토큰
                out = h if out.shape[1] == state["nt"] else h.transpose(0, 1)
            if state["store"]:
                rec["hid"].setdefault(i, []).append(bn(out, state["nt"]).detach().half())
            return out
        return f

    def blk_hook(i):
        def f(mod, inp, out):
            if state["store"]:
                rec["resid"].setdefault(i, []).append(bn(out, state["nt"]).norm(dim=-1).detach())
        return f

    def lnpre_hook(mod, inp, out):
        if state["mode"] != "register":
            return out
        o = bn(out, 197)
        o = torch.cat([o, torch.zeros_like(o[:, :1])], dim=1)  # 0 으로 시작하는 추가 토큰 1개
        return o if out.shape[1] == 197 else o.transpose(0, 1)

    hs = [b.mlp[1].register_forward_hook(hid_hook(i)) for i, b in enumerate(blocks)]
    hs += [b.register_forward_hook(blk_hook(i)) for i, b in enumerate(blocks)]
    hs.append(V.ln_pre.register_forward_hook(lnpre_hook))

    def run(xx, mode):
        state["mode"] = mode; state["nt"] = 198 if mode == "register" else 197
        rec["resid"].clear(); rec["hid"].clear()
        outs = []
        with torch.no_grad():
            for k in range(0, len(xx), 16):
                outs.append(model.encode_image(xx[k:k + 16]))
        e = torch.cat(outs); e = e / e.norm(dim=-1, keepdim=True)
        return e, {i: torch.cat(v) for i, v in rec["resid"].items()}, {i: torch.cat(v) for i, v in rec["hid"].items()}

    t0 = time.time()
    e0, res0, hid0 = run(x, "record")
    # 1) 이상치 토큰: 마지막 블록 잔차 노름이 이미지별 패치 중앙값의 4배 이상인 패치 토큰
    L = len(blocks) - 1
    norms = res0[L][:, 1:]
    med = norms.median(dim=1, keepdim=True).values
    out_mask = norms > 4 * med
    norm_by_layer = {i: {"patch_median": float(res0[i][:, 1:].median()), "patch_max": float(res0[i][:, 1:].max()),
                         "cls": float(res0[i][:, 0].mean())} for i in res0}
    first = next((i for i in range(len(blocks)) if (res0[i][:, 1:] > 4 * res0[i][:, 1:].median(dim=1, keepdim=True).values).any(dim=1).float().mean() > 0.5), None)
    print(f"[reg] 이상치 토큰: 이미지당 평균 {out_mask.sum(1).float().mean():.2f}개, 이미지 {(out_mask.any(1)).float().mean():.0%} 에 존재. 처음 나타나는 블록 {first}", flush=True)
    # 2) 레지스터 뉴런 후보: 블록 0..top_layer 의 MLP 뉴런, 이상치 위치 평균 활성 상위 top_k
    cand = []
    for i in range(a.top_layer + 1):
        h = hid0[i][:, 1:].float()                           # 패치만
        m = out_mask.unsqueeze(-1).float()
        avg_out = (h * m).sum((0, 1)) / m.sum().clamp(min=1)
        avg_all = h.mean((0, 1))
        for n in torch.argsort(-avg_out)[:a.top_k].tolist():
            cand.append((float(avg_out[n]), i, n, float(avg_all[n])))
    cand.sort(reverse=True)
    reg = cand[:a.top_k]
    # 3) 우리 hot 채널: 블록별 fc2 입력 |x| 채널 최댓값 상위 4 / 32 (모든 토큰)
    hot = {}
    for i in range(len(blocks)):
        c = hid0[i].float().abs().amax((0, 1))
        hot[i] = {"top4": torch.argsort(-c)[:4].tolist(), "top32": torch.argsort(-c)[:32].tolist(),
                  "absmax": float(c.max())}
        # 그 최댓값이 이상치 토큰에서 나오는 비율
    loc = {}
    for i in (5, 6, 7):
        h = hid0[i][:, 1:].float()
        top4 = hot[i]["top4"]
        sub = h[:, :, top4].abs()
        frac = float((sub * out_mask.unsqueeze(-1)).amax((0, 1)).min() / sub.amax((0, 1)).min()) if out_mask.any() else 0.0
        loc[i] = {"top4_max_on_outlier_tokens_ratio(min over 4ch)": round(frac, 3)}
    overlap = {i: sorted(set(hot[i]["top32"]) & {n for _, l, n, _ in reg if l == i}) for i in range(len(blocks))}
    reg_neurons = {}
    for _, l, n, _ in reg:
        reg_neurons.setdefault(l, []).append(n)
    # 4) 테스트 시 레지스터
    e1, res1, hid1 = (None, None, None)
    state["neurons"] = reg_neurons
    e1, res1, hid1 = run(x, "register")
    cos = (e0 * e1).sum(-1)

    def levels(h):
        h = h.float().flatten()
        amax = float(h.abs().max()); q = float(torch.quantile(h.abs()[::7].float(), 0.999))
        return amax, (255 * q / (2 * amax)) if amax > 0 else 0

    rng = {}
    for i in range(len(blocks)):
        a0, l0 = levels(hid0[i]); a1, l1 = levels(hid1[i][:, :-1])     # 레지스터 토큰 제외
        rng[i] = {"absmax_base": round(a0, 2), "bulk_levels_base": round(l0, 1),
                  "absmax_reg_excl_register": round(a1, 2), "bulk_levels_reg": round(l1, 1)}
    # 5) CIFAR 검증 앞 N장 zero-shot
    imgs, labels = C.load_eval_set("val")
    step = max(1, len(imgs) // a.cifar_n); idx = np.arange(0, len(imgs), step)[:a.cifar_n]
    xc = torch.from_numpy(np.stack([C.preprocess(Image.fromarray(imgs[k])) for k in idx]))
    t = torch.from_numpy(np.load(C.TEXT_EMB))
    accs = {}
    del hid0, hid1
    state["store"] = False
    for mode in ("record", "register"):
        ec, _, _ = run(xc[:a.cifar_n], mode)
        accs[mode] = float(((ec @ t.T).argmax(-1).numpy() == labels[idx]).mean())
    res = {"timestamp": C.now(), "args": vars(a), "n_images": len(x), "outlier_tokens_per_image": float(out_mask.sum(1).float().mean()),
           "first_outlier_block": first, "norm_by_layer": norm_by_layer,
           "register_neurons_top": [{"block": l, "neuron": n, "avg_on_outlier": round(v, 2), "avg_all": round(va, 3)} for v, l, n, va in reg],
           "our_hot_channels": hot, "overlap_with_top32": overlap, "hot_on_outlier_tokens": loc,
           "test_time_register": {"neurons": reg_neurons, "emb_cos_min": float(cos.min()), "emb_cos_mean": float(cos.mean()),
                                  "cifar_val_n": int(len(idx)), "cifar_top1_base": accs["record"], "cifar_top1_register": accs["register"]},
           "fc2_input_range": rng, "seconds": round(time.time() - t0)}
    (C.RESULTS / f"register_probe{a.tag}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    L2 = [f"# 레지스터 뉴런 진단 (E23) top_layer={a.top_layer} top_k={a.top_k}", "", f"- 이상치 토큰: 이미지당 {res['outlier_tokens_per_image']:.2f}개, 처음 나타나는 블록 {first}",
          "", "| 순위 | 블록 | 뉴런 | 이상치 토큰 평균 활성 | 전체 평균 | 우리 hot top4? | top32? |", "|---|---|---|---|---|---|---|"]
    for k, (v, l, n, va) in enumerate(reg, 1):
        L2.append(f"| {k} | {l} | {n} | {v:.2f} | {va:.3f} | {'예' if n in hot[l]['top4'] else ''} | {'예' if n in hot[l]['top32'] else ''} |")
    L2 += ["", f"- 테스트 시 레지스터: 임베딩 코사인 평균 {cos.mean():.4f} / 최소 {cos.min():.4f}, CIFAR 검증 {len(idx)}장 {accs['record']:.2%} -> {accs['register']:.2%}", "",
           "| 블록 | fc2 입력 absmax (원본) | 99.9% 눈금 칸 (원본) | absmax (레지스터, 이미지 토큰만) | 눈금 칸 (레지스터) |", "|---|---|---|---|---|"]
    for i in range(len(blocks)):
        r = rng[i]; L2.append(f"| {i} | {r['absmax_base']} | {r['bulk_levels_base']} | {r['absmax_reg_excl_register']} | {r['bulk_levels_reg']} |")
    (C.RESULTS / f"register_probe{a.tag}.md").write_text("\n".join(L2) + "\n")
    print("\n".join(L2))
    for h in hs:
        h.remove()


if __name__ == "__main__":
    main()
