"""E44: 이미지·텍스트 섞는 비율(α)을 따로 + 체크포인트 평균(ep1·ep2). 재학습 없음, FP32 뒤쪽만 계산.

  python -m src.da_grid --tag e37 --out e44

- 원천: ep2 (E37 마지막), soup = (ep1 + ep2)/2
- 격자: α_img, α_txt ∈ {0.5, 0.625, 0.75, 0.875, 1.0} × 원천 2 = 50개
- 선택(사전 등록): CIFAR val ≥ 67.80 (α=0 의 68.80 −1.0, E36/E37 과 같은 제약) 중 select R@1 최고
- 선택 가중치 → models_da/da2_chosen_{out}.pt (da_text export --tag {out} 로 변환)
"""
from __future__ import annotations
import argparse, json, time
import numpy as np
from . import common as C
from . import da_common as D
from .da_text import make_text_tail, open_t


def main():
    import torch
    from .coco_retrieval import captions_for, per_image, ranks
    from .da_sweep import embed
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="e37"); ap.add_argument("--out", default="e44")
    a = ap.parse_args(); torch.set_num_threads(16)
    th0 = torch.load(D.DA_DIR / "da2_init.pt")
    e1 = torch.load(D.DA_DIR / f"da2_{a.tag}_ep1.pt"); e2 = torch.load(D.DA_DIR / f"da2_{a.tag}_ep2.pt")
    src = {"ep2": e2, "soup": {k: {x: 0.5 * (e1[k][x] + e2[k][x]) for x in e2[k]} for k in e2}}
    m = D.load_clip(); it, tt = D.make_tail(m.visual).eval(), make_text_tail(m).eval()
    fs, _ = D.open_feat("select"); fc, _ = D.open_feat("cifarval"); TS, ES = open_t("tselect"); TC, EC = open_t("tcifar")
    _, img_of, _ = captions_for("select"); _, lab = C.load_eval_set("val"); nimg = int(img_of.max() + 1)
    fs = np.asarray(fs, dtype=np.float32); fc_n = len(fc)

    def temb(T, E):
        out = []
        with torch.no_grad():
            for s in range(0, len(T), 512):
                e = tt(torch.from_numpy(np.asarray(T[s:s + 512], dtype=np.float32)), torch.from_numpy(E[s:s + 512])).numpy()
                out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        return np.concatenate(out)

    grid = [0.5, 0.625, 0.75, 0.875, 1.0]
    mix = lambda S, k, al: {x: (1 - al) * th0[k][x] + al * S[k][x] for x in th0[k]}
    rows, t0 = [], time.time()
    for sname, S in src.items():
        img_cache = {}
        for ai in grid:                                  # 이미지 임베딩은 α_img 마다 한 번
            it.load_state_dict(mix(S, "img", ai)); img_cache[ai] = (embed(it, fs), embed(it, fc))
        for at in grid:
            tt.load_state_dict(mix(S, "txt", at)); txs, txc = temb(TS, ES), temb(TC, EC)
            for ai in grid:
                es, ec = img_cache[ai]; r = ranks(es, txs, img_of)
                row = {"src": sname, "a_img": ai, "a_txt": at,
                       **{f"R@{k}": float(per_image(r, img_of, nimg, k).mean() * 100) for k in (1, 5, 10)},
                       "cifar_val": float((np.argmax(ec @ txc.T, 1) == lab).mean() * 100)}
                rows.append(row)
                print(f"[e44] {sname} α_img={ai} α_txt={at}: select R@1 {row['R@1']:.2f} / CIFAR val {row['cifar_val']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    ok = [r for r in rows if r["cifar_val"] >= 67.80]
    best = max(ok, key=lambda r: r["R@1"])
    ref = [r for r in rows if r["src"] == "ep2" and r["a_img"] == 0.75 and r["a_txt"] == 0.75][0]
    res = {"rows": rows, "chosen": best, "ref_e37": ref}
    (C.RESULTS / f"da_grid_{a.out}.json").write_text(json.dumps(res, indent=2))
    L = [f"# {a.out}: α_img × α_txt × (ep2, ep1·ep2 평균) — COCO-select R@1 / CIFAR val (FP32, 재학습 없음)", "",
         "| 원천 | α_img \\ α_txt | " + " | ".join(str(g) for g in grid) + " |", "|---|---|" + "---|" * len(grid)]
    for sname in src:
        for ai in grid:
            cells = [next(r for r in rows if r["src"] == sname and r["a_img"] == ai and r["a_txt"] == at) for at in grid]
            L.append(f"| {sname} | {ai} | " + " | ".join(f"{c['R@1']:.2f} / {c['cifar_val']:.2f}" for c in cells) + " |")
    L += ["", f"- 기준(E37 선택, ep2 0.75/0.75): select {ref['R@1']:.2f} / CIFAR {ref['cifar_val']:.2f}",
          f"- 선택(CIFAR val ≥ 67.80 중 select 최고): {best['src']} α_img={best['a_img']} α_txt={best['a_txt']} → select {best['R@1']:.2f} / CIFAR {best['cifar_val']:.2f}"]
    (C.RESULTS / f"da_grid_{a.out}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    S = src[best["src"]]
    torch.save({"img": mix(S, "img", best["a_img"]), "txt": mix(S, "txt", best["a_txt"])}, D.DA_DIR / f"da2_chosen_{a.out}.pt")


if __name__ == "__main__":
    main()
