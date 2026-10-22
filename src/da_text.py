"""E36: 텍스트 인코더 뒤쪽 4블록도 함께 학습 (이미지 쪽은 E33 과 같은 뒤쪽 4블록).

텍스트 인코더는 호스트 CPU 에서 FP32 로 돌고 양자화하지 않으므로, 학습해도 이미지 쪽 양자화 처방(SPL)과 무관하다.
텍스트도 블록 0~7 출력을 저장해 두고 뒤쪽(블록 8~11 + ln_final + text_projection)만 학습한다.

  python -m src.da_text check                         # 텍스트 앞/뒤 나눔 = encode_text 확인
  python -m src.da_text cache --n 30000               # 학습 캡션 텍스트 앞부분 저장 (+ select 캡션, CIFAR 프롬프트)
  python -m src.da_text train --n 30000 --tag e36 [--epochs 2]
  python -m src.da_text sweep --tag e36 --ckpt da2_e36_ep2.pt
  HALIO_MODELS_SUBDIR=models_da2 python -m src.da_text export --tag e36   # 이미지 FP32 ONNX(P1) + 적응 텍스트 임베딩 파일
"""
from __future__ import annotations

import json, math, resource, shutil, sys, time
import numpy as np
from . import common as C
from . import da_common as D

SPLIT_T = 8


def text_front(m, tok):
    import torch
    x = m.token_embedding(tok) + m.positional_embedding
    bf = getattr(m.transformer, "batch_first", True)
    if not bf: x = x.transpose(0, 1)
    for b in m.transformer.resblocks[:SPLIT_T]:
        x = b(x, attn_mask=m.attn_mask)
    if not bf: x = x.transpose(0, 1)
    return x, tok.argmax(dim=-1)


def make_text_tail(m):
    import torch.nn as nn

    class TTail(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(list(m.transformer.resblocks[SPLIT_T:]))
            self.ln_final = m.ln_final
            self.text_projection = m.text_projection
            self._m = [m]

        def forward(self, x, eot):
            import torch
            mm = self._m[0]; bf = getattr(mm.transformer, "batch_first", True)
            if not bf: x = x.transpose(0, 1)
            for b in self.blocks:
                x = b(x, attn_mask=mm.attn_mask)
            if not bf: x = x.transpose(0, 1)
            x = self.ln_final(x)[torch.arange(x.shape[0]), eot]
            tp = self.text_projection
            return tp(x) if isinstance(tp, nn.Linear) else x @ tp
    return TTail()


def tok():
    import open_clip
    return open_clip.get_tokenizer(C.MODEL_NAME)


def cmd_check():
    import torch
    m = D.load_clip(); tt = make_text_tail(m); T = tok()
    caps = ["a photo of a dog", "A man in an orange hat starring at something.", "two people riding bikes on a city street at night"]
    t = T(caps)
    with torch.no_grad():
        a = m.encode_text(t).numpy(); f, e = text_front(m, t); b = tt(f, e).numpy(); b16 = tt(f.half().float(), e).numpy()
    err = float(np.abs(a - b).max()); cos16 = float(np.min(np.sum(a * b16, 1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b16, axis=1)))
    print(f"[da-text] 텍스트 앞/뒤 나눔 = encode_text: 최대 오차 {err:.2e}, fp16 저장 최소 코사인 {cos16:.6f}, 앞부분 최대 |x| {float(f.abs().max()):.1f}")
    if err > 1e-4 or cos16 < 0.9999: sys.exit(1)


def _cache_texts(m, texts, name, bs=512, prefix=None):
    import torch
    T = tok(); n = len(texts)
    p = D.FEAT_DIR / f"{name}.f16"
    if p.exists():
        print(f"[da-text] {name} 이미 있음"); return
    out = np.memmap(str(p) + ".part", dtype=np.float16, mode="w+", shape=(n, 77, 512)); eot = np.empty(n, dtype=np.int64)
    t0 = time.time(); s_start = 0
    if prefix:                                              # E37: 앞부분(E36 저장분)과 같은 캡션 → 복사 후 16개 다시 계산해 확인
        old, oe = open_t(prefix[0]); k = prefix[1]
        for s in range(0, k, 8192):
            out[s:min(k, s + 8192)] = old[s:min(k, s + 8192)]
        eot[:k] = oe
        chk = np.sort(np.random.default_rng(0).choice(k, 16, replace=False))
        with torch.no_grad():
            f, e = text_front(m, T([texts[i] for i in chk]))
        err = float(np.abs(f.numpy().astype(np.float16).astype(np.float32) - np.asarray(old[chk], dtype=np.float32)).max())
        assert np.array_equal(e.numpy(), oe[chk]) and err < 0.5, f"재사용 확인 실패 {err}"
        print(f"[da-text] 앞 {k}개 재사용 (다시 계산 16개 최대 차 {err:.2e})", flush=True); s_start = k
    with torch.no_grad():
        for s in range(s_start, n, bs):
            f, e = text_front(m, T(texts[s:s + bs])); out[s:s + len(e)] = f.numpy().astype(np.float16); eot[s:s + len(e)] = e.numpy()
            if (s // bs) % 40 == 0: print(f"[da-text] {name} {s + len(e)}/{n} ({time.time() - t0:.0f}s)", flush=True)
    out.flush(); del out
    import os; os.replace(str(p) + ".part", p); np.save(D.FEAT_DIR / f"{name}_eot.npy", eot)
    (D.FEAT_DIR / f"{name}.json").write_text(json.dumps({"n": n, "shape": [n, 77, 512], "dtype": "float16"}))
    print(f"[da-text] {name}: {n}개 {time.time() - t0:.0f}s, {p.stat().st_size / 1e9:.2f} GB", flush=True)


def cmd_cache(n):
    import torch
    torch.set_num_threads(16)
    from .da_data import captions
    from .coco_retrieval import captions_for
    m = D.load_clip()
    ann = captions(); files = D.read_list(D.train_list(n))[:n]
    idx = {fn: k for k, fn in enumerate(files)}; fid = {im["id"]: im["file_name"] for im in ann["images"]}
    caps = sorted((idx[fid[c["image_id"]]], c["id"], c["caption"].strip()) for c in ann["annotations"] if fid[c["image_id"]] in idx)
    ci_p = D.FEAT_DIR / f"train_{n}_text_img.npy"
    if not ci_p.exists():                                   # E37: 캡션→사진 번호만 저장 (E33 고정 텍스트 임베딩은 안 씀)
        np.save(ci_p, np.array([c[0] for c in caps], dtype=np.int64))
    ci = np.load(ci_p)
    assert np.array_equal(ci, np.array([c[0] for c in caps])), "캡션 순서가 저장분과 다름"
    pre = None
    if n > 30000 and (D.FEAT_DIR / "ttrain_30000.json").exists():
        c30 = np.load(D.FEAT_DIR / "train_30000_text_img.npy"); k = len(c30)
        assert np.array_equal(c30, ci[:k]), "앞 30,000장 캡션 순서가 다름"
        pre = ("ttrain_30000", k)
    _cache_texts(m, [c[2] for c in caps], f"ttrain_{n}", prefix=pre)
    _, _, sc = captions_for("select"); _cache_texts(m, sc, "tselect")
    names = C.cifar100_class_names(); _cache_texts(m, [C.PROMPT_TEMPLATE.format(x) for x in names], "tcifar")


def open_t(name):
    meta = json.loads((D.FEAT_DIR / f"{name}.json").read_text())
    return np.memmap(D.FEAT_DIR / f"{name}.f16", dtype=np.float16, mode="r", shape=tuple(meta["shape"])), np.load(D.FEAT_DIR / f"{name}_eot.npy")


def cmd_train(a):
    import torch, torch.nn.functional as F
    torch.manual_seed(a.seed); torch.set_num_threads(16); rng = np.random.default_rng(a.seed)
    feat, _ = D.open_feat(f"train_{a.n}"); TF, TE = open_t(f"ttrain_{a.n}")
    cimg = np.load(D.FEAT_DIR / f"train_{a.n}_text_img.npy")
    start = np.searchsorted(cimg, np.arange(a.n)); cnt = np.bincount(cimg, minlength=a.n)
    m = D.load_clip()
    for p in m.parameters(): p.requires_grad_(False)
    it, tt = D.make_tail(m.visual), make_text_tail(m)
    params = list(it.parameters()) + list(tt.parameters())
    for p in params: p.requires_grad_(True)
    scale = float(m.logit_scale.exp())
    init = D.DA_DIR / "da2_init.pt"
    if not init.exists():
        torch.save({"img": D.tail_state(it), "txt": D.tail_state(tt)}, init)
    total = (a.n // a.batch) * a.epochs
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.1, betas=(0.9, 0.98), eps=1e-6)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / 100 if s < 100 else 0.5 * (1 + math.cos(math.pi * (s - 100) / max(1, total - 100))))
    it.train(); tt.train(); step, t0, log = 0, time.time(), []
    print(f"[da-text] 사진 {a.n}, 에폭 {a.epochs} ({total}스텝), 학습 파라미터 {sum(p.numel() for p in params) / 1e6:.1f}M", flush=True)
    for ep in range(a.epochs):
        rl, rn = 0.0, 0
        for ci in rng.permutation((a.n + 4095) // 4096):
            s0, s1 = ci * 4096, min(a.n, (ci + 1) * 4096)
            X = torch.from_numpy(np.asarray(feat[s0:s1], dtype=np.float32)); perm = rng.permutation(s1 - s0)
            for b in range(0, len(perm) - a.batch + 1, a.batch):
                ids = perm[b:b + a.batch]; g = s0 + ids; cap = start[g] + rng.integers(0, cnt[g])
                o = np.argsort(cap); cs = cap[o]                       # memmap 은 정렬된 순서로 읽는다
                tf = torch.from_numpy(np.asarray(TF[cs], dtype=np.float32)); inv = np.argsort(o)
                ts = time.time()
                img = F.normalize(it(X[ids]), dim=-1)
                txt = F.normalize(tt(tf, torch.from_numpy(TE[cs])), dim=-1)[inv]
                lg = scale * img @ txt.T; lab = torch.arange(len(ids))
                loss = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); step += 1
                rl += loss.item(); rn += 1
                if step % 20 == 0 or step <= 3:
                    el = time.time() - t0
                    print(f"[da-text] ep{ep} step {step}/{total} loss {rl / rn:.4f} {(time.time() - ts) * 1000:.0f} ms/step, 남은 {el / step * (total - step) / 60:.0f}분, 메모리 {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)
                    log.append({"step": step, "loss": rl / rn}); rl, rn = 0.0, 0
        torch.save({"img": D.tail_state(it), "txt": D.tail_state(tt)}, D.DA_DIR / f"da2_{a.tag}_ep{ep + 1}.pt")
        print(f"[da-text] 에폭 {ep + 1} 저장", flush=True)
    (C.LOGS / f"da_text_log_{a.tag}.json").write_text(json.dumps({"args": vars(a), "minutes": (time.time() - t0) / 60, "log": log}, indent=2))


def cmd_sweep(a):
    import torch
    from .coco_retrieval import captions_for, per_image, ranks
    from .da_sweep import embed
    torch.set_num_threads(16)
    th0 = torch.load(D.DA_DIR / "da2_init.pt"); th1 = torch.load(D.DA_DIR / a.ckpt)
    m = D.load_clip(); it, tt = D.make_tail(m.visual).eval(), make_text_tail(m).eval()
    fs, _ = D.open_feat("select"); fc, _ = D.open_feat("cifarval"); TS, ES = open_t("tselect"); TC, EC = open_t("tcifar")
    _, img_of, _ = captions_for("select"); _, lab = C.load_eval_set("val"); nimg = int(img_of.max() + 1)

    def temb(T, E):
        out = []
        with torch.no_grad():
            for s in range(0, len(T), 512):
                e = tt(torch.from_numpy(np.asarray(T[s:s + 512], dtype=np.float32)), torch.from_numpy(E[s:s + 512])).numpy()
                out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        return np.concatenate(out)
    rows = []
    for al in [0.0, 0.25, 0.5, 0.75, 1.0]:
        mix = lambda k: {x: (1 - al) * th0[k][x] + al * th1[k][x] for x in th0[k]}
        it.load_state_dict(mix("img")); tt.load_state_dict(mix("txt"))
        es, ec = embed(it, fs), embed(it, fc); txs, txc = temb(TS, ES), temb(TC, EC)
        r = ranks(es, txs, img_of)
        row = {"alpha": al, **{f"R@{k}": float(per_image(r, img_of, nimg, k).mean() * 100) for k in (1, 5, 10)},
               "cifar_val": float((np.argmax(ec @ txc.T, 1) == lab).mean() * 100)}
        rows.append(row); print(f"[da-text-sweep] α={al}: select R@1 {row['R@1']:.2f} R@5 {row['R@5']:.2f} / CIFAR val {row['cifar_val']:.2f}", flush=True)
    base = rows[0]; ok = [r for r in rows[1:] if r["cifar_val"] >= base["cifar_val"] - 1.0]
    best = max(ok, key=lambda r: r["R@1"]) if ok else None
    res = {"ckpt": a.ckpt, "rows": rows, "chosen_alpha": best["alpha"] if best else None}
    (C.RESULTS / f"da_sweep_{a.tag}.json").write_text(json.dumps(res, indent=2))
    L = [f"# {a.tag}: 이미지+텍스트 뒤쪽 4블록 학습, α 탐색 (COCO-select, CIFAR val)", "", "| α | select R@1 | R@5 | R@10 | CIFAR val |", "|---|---|---|---|---|"]
    L += [f"| {r['alpha']} | {r['R@1']:.2f} | {r['R@5']:.2f} | {r['R@10']:.2f} | {r['cifar_val']:.2f} |" for r in rows]
    L += ["", f"- 선택 α = {res['chosen_alpha']} (CIFAR val −1%p 이내 중 select R@1 최고). 비교: E33 select R@1 53.99"]
    (C.RESULTS / f"da_sweep_{a.tag}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    if best:
        torch.save({k: {x: (1 - best["alpha"]) * th0[k][x] + best["alpha"] * th1[k][x] for x in th0[k]} for k in th0}, D.DA_DIR / f"da2_chosen_{a.tag}.pt")


def full_text_state(m, tt):
    """텍스트 꼬리 상태를 CLIP 전체 키 이름으로 (open_clip load_state_dict(strict=False) 용)."""
    out = {}
    for k, v in tt.state_dict().items():
        if k.startswith("blocks."):
            i, rest = k.split(".", 2)[1:]; out[f"transformer.resblocks.{SPLIT_T + int(i)}.{rest}"] = v.clone()
        else:
            out[k] = v.clone()
    return out


def cmd_export(a):
    import torch, onnx, collections
    from PIL import Image
    from .export_onnx import export, fold_constants, parity, text_embeddings
    from .coco_retrieval import captions_for, text_path
    assert C.MODELS.name.startswith("models_da"), "HALIO_MODELS_SUBDIR=models_da2(또는 models_da3) 로 실행"
    C.ensure_dirs()
    st = torch.load(D.DA_DIR / f"da2_chosen_{a.tag}.pt")
    m = D.load_clip(); it, tt = D.make_tail(m.visual), make_text_tail(m)
    it.load_state_dict(st["img"]); tt.load_state_dict(st["txt"]); m.eval()
    torch.save(full_text_state(m, tt), C.MODELS / "text_state.pt")
    info = export(m, C.RAW_ONNX); info.update(fold_constants(C.RAW_ONNX, C.FP32_ONNX))
    data, _ = C.load_cifar100("train"); val = C.read_index_list(C.SPLITS / "val.txt")[:20]
    par = parity(m, C.FP32_ONNX, np.stack([C.preprocess(Image.fromarray(data[i])) for i in val]))
    names = C.cifar100_class_names(); np.save(C.TEXT_EMB, text_embeddings(m, tok(), names))
    shutil.copy2(C.STORE / "models" / "cifar100_classes.json", C.MODELS / "cifar100_classes.json")
    T = tok()
    for split in ("select", "confirm"):
        _, _, caps = captions_for(split); out = []
        with torch.no_grad():
            for i in range(0, len(caps), 256):
                e = m.encode_text(T(caps[i:i + 256])).float().numpy(); out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        np.save(text_path(split), np.concatenate(out).astype(np.float32))
    ops = lambda p: collections.Counter(n.op_type for n in onnx.load(str(p), load_external_data=False).graph.node)
    same = ops(C.FP32_ONNX) == ops(C.STORE / "models" / C.FP32_ONNX.name)
    (C.LOGS / f"p1_parity_{a.tag}.json").write_text(json.dumps({"export": info, "parity": par, "same_ops": same}, indent=2, default=str))
    print(f"[da-text-export] P1 {par['max_abs_err']:.2e}/{par['min_cosine']:.6f} {'통과' if par['passed'] else '실패'}, 연산 구성 동일 {same}, 텍스트 임베딩(적응) 저장")
    if not (par["passed"] and same): sys.exit(1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("cmd")
    ap.add_argument("--n", type=int, default=30000); ap.add_argument("--tag", default="e36"); ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    {"check": cmd_check, "cache": lambda: cmd_cache(a.n), "train": lambda: cmd_train(a), "sweep": lambda: cmd_sweep(a), "export": lambda: cmd_export(a)}[a.cmd]()
