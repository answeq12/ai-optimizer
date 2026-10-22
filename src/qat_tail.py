"""E39: 뒤쪽 4블록 양자화 보정 학습 (QAT + FP32 증류). 정답 = 같은 모델의 FP32 출력 (캡션·라벨 안 씀)

아이디어: 배포 INT8 모델(SPL)의 블록 7 출력에는 블록 0~7 양자화 오차가 이미 섞여 있다. 그 '오차 섞인 입력'을 받아,
뒤쪽 4블록(8~11)+ln_post+proj 를 INT8 흉내(fake-quant: 가중치 채널별 int8, 활성 텐서별 uint8 MinMax) 상태로 학습해
FP32 원본의 최종 임베딩과 같아지게 한다. 학습 후 FP32 가중치를 원래와 같은 SPL 처방으로 ORT 양자화한다.

  python -m src.qat_tail noisy --base orig --n 10000     # INT8 모델에서 블록 7 출력 저장 (ORT, 실제 양자화 오차)
  python -m src.qat_tail train --base orig --n 10000     # 보정 학습 → models_da/qat_{base}_tail.pt
  HALIO_MODELS_SUBDIR=models_qat_orig python -m src.qat_tail export --base orig
base: orig = 원본 CLIP (teacher = 원본 FP32), da = E33 적응 모델 (teacher = 적응 FP32)
"""
from __future__ import annotations
import json, math, sys, time
import numpy as np
from PIL import Image
from . import common as C
from . import da_common as D

INT8 = {"orig": C.STORE / "models" / "imp" / "SPL-b567-k32.onnx", "da": C.STORE / "models_da" / "imp" / "SPL-b567-k32.onnx"}
FP32 = {"orig": C.STORE / "models" / "clip_vitb16_image_fp32.onnx", "da": C.STORE / "models_da" / "clip_vitb16_image_fp32.onnx"}


def load_base(base):
    import torch
    m = D.load_clip()
    if base == "da":
        D.make_tail(m.visual).load_state_dict(torch.load(D.DA_DIR / "da_tail_chosen.pt"))
    return m


def block7_name(model_path):
    import onnx
    g = onnx.load(str(model_path), load_external_data=False).graph
    cand = [n for n in g.node if n.op_type == "LayerNormalization" and "resblocks.8/" in n.name and "ln_1" in n.name]
    if len(cand) != 1:
        sys.exit(f"블록 8 ln_1 노드를 하나로 찾지 못함: {[n.name for n in cand]}")
    return cand[0].input[0]


def session_with_output(model_path, name, threads=16):
    import onnx, onnxruntime as ort
    m = onnx.load(str(model_path))
    m.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    return ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])


def cmd_noisy(base, n):
    import torch
    name = block7_name(INT8[base]); print(f"[qat] 블록 7 출력 텐서: {name}", flush=True)
    # 같은 이름이 FP32 ONNX 에서도 torch front() 와 일치하는지 확인 (텐서가 맞는지 검증)
    fsess = session_with_output(FP32[base], block7_name(FP32[base]))
    m = load_base(base)
    files = D.read_list(D.TRAIN_LIST)[:n]
    x = C.preprocess(Image.open(D.TRAIN_DIR / files[0]))[None]
    with torch.no_grad():
        ref = D.front(m.visual, torch.from_numpy(x)).numpy()
    got = fsess.run([block7_name(FP32[base])], {fsess.get_inputs()[0].name: x})[0]
    err = float(np.abs(ref - got.reshape(ref.shape)).max()); rel = err / float(np.abs(ref).max())
    print(f"[qat] FP32 ONNX 블록7 = torch front: 최대 오차 {err:.2e} (최대값 대비 {rel:.1e})", flush=True)
    if rel > 1e-4: sys.exit("블록 7 텐서가 맞지 않음")   # 값 크기가 최대 약 86 이라 절대 오차 대신 상대 오차로 판정
    sess = session_with_output(INT8[base], name); inp = sess.get_inputs()[0].name
    p = D.FEAT_DIR / f"noisy7_{base}_{n}.f16"
    if p.exists(): print("[qat] 이미 있음"); return
    out = np.memmap(str(p) + ".part", dtype=np.float16, mode="w+", shape=(n, 197, 768)); t0 = time.time()
    for i, f in enumerate(files):
        x = C.preprocess(Image.open(D.TRAIN_DIR / f))[None]
        out[i] = sess.run([name], {inp: x})[0].reshape(197, 768).astype(np.float16)
        if (i + 1) % 1000 == 0: print(f"[qat] noisy {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    out.flush(); del out
    import os; os.replace(str(p) + ".part", p)
    D.write_meta(f"noisy7_{base}_{n}", n, {"model": str(INT8[base]), "tensor": name})
    clean, _ = D.open_feat(f"train_30000"); nz, _ = D.open_feat(f"noisy7_{base}_{n}")
    a, b = np.asarray(clean[:200], np.float32), np.asarray(nz[:200], np.float32)
    rel = float(np.linalg.norm(a - b) / np.linalg.norm(a))
    print(f"[qat] 블록7 양자화 오차(앞 200장, base={base}): 상대 오차 {rel:.4f}  ※ base=da 면 clean 저장분이 원본 front 와 같다(블록 0~7 동일)", flush=True)


# ---- fake-quant 꼬리 ----------------------------------------------------------
def fq_w(W):                       # nn.Linear 가중치 [out,in]: 출력 채널별 대칭 int8 (ORT per-channel QDQ 와 같은 축)
    s = W.detach().abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    return W + ((W / s).round().clamp(-127, 127) * s - W).detach()


def fq_a(x, rng_):                 # 활성: 텐서별 비대칭 uint8, 범위 고정(캘리브 MinMax, 0 포함)
    lo, hi = rng_
    s = max(hi - lo, 1e-8) / 255.0; z = round(-lo / s)
    q = ((x / s).round() + z).clamp(0, 255)
    return x + (((q - z) * s) - x).detach()


def tail_forward(v, blocks, x, R=None, rec=None):
    """open_clip 블록을 명시적으로 계산 (가중치 MatMul 4종 + proj 를 INT8 흉내). R=None 이면 FP32, rec 이면 범위 기록"""
    import torch, torch.nn.functional as F

    def act(name, t):
        if rec is not None:
            lo, hi = float(t.min()), float(t.max()); o = rec.get(name, (0.0, 0.0)); rec[name] = (min(o[0], lo, 0.0), max(o[1], hi, 0.0))
        return fq_a(t, R[name]) if R is not None else t
    W = (lambda w: fq_w(w)) if R is not None else (lambda w: w)
    for i, b in enumerate(blocks):
        a = b.attn; N, L, Dm = x.shape; H = a.num_heads
        h = b.ln_1(x)
        qkv = F.linear(act(f"{i}.qkv", h), W(a.in_proj_weight), a.in_proj_bias)
        q, k, vv = qkv.split(Dm, dim=-1)
        q, k, vv = (t.reshape(N, L, H, Dm // H).transpose(1, 2) for t in (q, k, vv))
        o = F.scaled_dot_product_attention(q, k, vv).transpose(1, 2).reshape(N, L, Dm)
        x = x + F.linear(act(f"{i}.out", o), W(a.out_proj.weight), a.out_proj.bias)
        h2 = b.ln_2(x)
        f1 = b.mlp.gelu(F.linear(act(f"{i}.fc1", h2), W(b.mlp.c_fc.weight), b.mlp.c_fc.bias))
        x = x + F.linear(act(f"{i}.fc2", f1), W(b.mlp.c_proj.weight), b.mlp.c_proj.bias)
    pooled = v.ln_post(x)[:, 0]
    P = v.proj
    if R is not None:
        Pq = P + (((P / (P.detach().abs().amax(0, keepdim=True).clamp(min=1e-8) / 127)).round().clamp(-127, 127)
                   * (P.detach().abs().amax(0, keepdim=True).clamp(min=1e-8) / 127)) - P).detach()
        return act("proj", pooled) @ Pq
    if rec is not None: act("proj", pooled)
    return pooled @ P


def cmd_train(base, n, a):
    import torch, torch.nn.functional as F
    torch.manual_seed(0); torch.set_num_threads(16); rng = np.random.default_rng(0)
    m = load_base(base); v = m.visual
    for p in m.parameters(): p.requires_grad_(False)
    blocks = list(v.transformer.resblocks[D.SPLIT_AT:])
    # 0) 명시 계산 = open_clip 계산 확인
    clean, _ = D.open_feat("train_30000"); nz, _ = D.open_feat(f"noisy7_{base}_{n}")
    x0 = torch.from_numpy(np.asarray(clean[:4], np.float32))
    with torch.no_grad():
        e_ref = D.make_tail(v)(x0); e_me = tail_forward(v, blocks, x0)
    err = float((e_ref - e_me).abs().max()); print(f"[qat] 명시 계산 = open_clip: 최대 오차 {err:.2e}", flush=True)
    if err > 1e-3 * float(e_ref.abs().max()): sys.exit(1)
    # 1) 활성 범위: 캘리브 64장(ORT 와 같은 목록)의 FP32 값으로 MinMax
    cal = C.load_calib_images(C.SPLITS / "calib_coco_64_s0.txt"); rec = {}
    with torch.no_grad():
        for i in range(0, len(cal), 16):
            tail_forward(v, blocks, D.front(v, torch.from_numpy(cal[i:i + 16])), rec=rec)
    # 2) teacher = 같은 모델 FP32 (깨끗한 입력)
    T = []
    with torch.no_grad():
        for s in range(0, n, 256):
            e = tail_forward(v, blocks, torch.from_numpy(np.asarray(clean[s:min(n, s + 256)], np.float32))); T.append(F.normalize(e, dim=-1))
    T = torch.cat(T)
    def score(k=500):
        with torch.no_grad():
            s = F.normalize(tail_forward(v, blocks, torch.from_numpy(np.asarray(nz[n - k:n], np.float32)), R=rec), dim=-1)
            return float((s * T[n - k:n]).sum(1).mean())
    c0 = score(); print(f"[qat] 학습 전: INT8 흉내 출력과 FP32 의 평균 코사인(보류 500장) {c0:.5f}", flush=True)
    params = [p for b in blocks for p in b.parameters()] + list(v.ln_post.parameters()) + [v.proj]
    for p in params: p.requires_grad_(True)
    tr = n - 500; steps = (tr // a.batch) * a.epochs
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / max(1, steps))))
    t0, st = time.time(), 0
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        for b0 in range(0, tr - a.batch + 1, a.batch):
            idx = np.sort(perm[b0:b0 + a.batch])
            xs = torch.from_numpy(np.asarray(nz[idx], np.float32))
            s = F.normalize(tail_forward(v, blocks, xs, R=rec), dim=-1)
            loss = (1 - (s * T[idx]).sum(1)).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); st += 1
            if st % 25 == 0 or st <= 2:
                print(f"[qat] ep{ep} step {st}/{steps} 1-cos {loss.item():.5f} 경과 {(time.time() - t0) / 60:.1f}분", flush=True)
    c1 = score(); print(f"[qat] 학습 후: 평균 코사인(보류 500장) {c0:.5f} → {c1:.5f}", flush=True)
    torch.save(D.tail_state(D.make_tail(v)), D.DA_DIR / f"qat_{base}_tail.pt")
    (C.LOGS / f"qat_{base}.json").write_text(json.dumps({"n": n, "args": vars(a), "cos_before": c0, "cos_after": c1,
                                                          "minutes": (time.time() - t0) / 60, "ranges": rec}, indent=2))


def cmd_export(base):
    import torch, shutil, collections, onnx
    from .export_onnx import export, fold_constants, parity
    assert C.MODELS.name == f"models_qat_{base}", f"HALIO_MODELS_SUBDIR=models_qat_{base} 로 실행"
    C.ensure_dirs(); m = load_base(base)
    D.make_tail(m.visual).load_state_dict(torch.load(D.DA_DIR / f"qat_{base}_tail.pt")); m.eval()
    info = export(m, C.RAW_ONNX); info.update(fold_constants(C.RAW_ONNX, C.FP32_ONNX))
    data, _ = C.load_cifar100("train"); val = C.read_index_list(C.SPLITS / "val.txt")[:20]
    par = parity(m, C.FP32_ONNX, np.stack([C.preprocess(Image.fromarray(data[i])) for i in val]))
    src = C.STORE / "models"
    for f in ["text_emb_cifar100.npy", "cifar100_classes.json"] + sorted(p.name for p in src.glob("coco_ret_text_*")):
        shutil.copy2(src / f, C.MODELS / f)
    ops = lambda p: collections.Counter(nd.op_type for nd in onnx.load(str(p), load_external_data=False).graph.node)
    same = ops(C.FP32_ONNX) == ops(src / C.FP32_ONNX.name)
    print(f"[qat-export] P1 {par['max_abs_err']:.2e}/{par['min_cosine']:.6f} {'통과' if par['passed'] else '실패'}, 연산 구성 동일 {same}")
    if not (par["passed"] and same): sys.exit(1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--base", default="orig")
    ap.add_argument("--n", type=int, default=10000); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    {"noisy": lambda: cmd_noisy(a.base, a.n), "train": lambda: cmd_train(a.base, a.n, a), "export": lambda: cmd_export(a.base)}[a.cmd]()
