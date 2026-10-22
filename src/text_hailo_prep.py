"""Hailo 텍스트 인코더 준비 (파일럿 환경 ~/venvs/halio 에서 실행. open_clip·onnxruntime 필요).

  python -m src.text_hailo_prep

하는 일
 1) Model Zoo 설정의 공식 텍스트 인코더 ONNX(clip_text_encoder_vitb_16_sim.onnx)를 내려받는다
 2) ONNX 구조 확인: 입력 형식, 시작 노드(Add_1)·끝 노드(Add_1515)의 입력과 그 뒤에 남는 연산
 3) Hailo 에 넣을 입력(77x512)이 무엇인지 open_clip 과 비교해 정한다
    (토큰 임베딩만인지, 위치 임베딩까지 더한 것인지) -> ORT 로 Add_1515 출력이 open_clip 트랜스포머 출력과 같은 쪽
 4) 칩 밖 후처리(ln_final -> EOT 위치 선택 -> text_projection)를 numpy 로 재현해 open_clip encode_text 와 비교
 5) 캘리브레이션 입력: COCO 검색 '제외' 목록(splits/coco_ret_exclude.txt) 이미지의 캡션만 사용
    (select/confirm 평가와 겹치지 않는다. PROTOCOL 의 캘리브레이션 규칙과 같은 원칙)
 6) 간이 확인용: select 캡션 500개(일정 간격)의 입력과 FP32 기준 임베딩. 평가 결론용이 아니다

결과: <STORE>/hailo/text/prep.npz, logs/dfc/text_prep.json
"""
from __future__ import annotations

import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from src import common as C
from src import coco_retrieval as R

ZOO_NET = "clip_vit_b_16_text_encoder"
OUT = C.STORE / "hailo" / "text"
LOG = C.LOGS / "dfc" / "text_prep.json"
N_CHECK = 500


def zoo_read(suffix: str) -> str:
    import glob
    c = sorted(glob.glob(str(C.ROOT / "installers" / "hailo_model_zoo-*.whl"))) or \
        sorted(glob.glob(str(C.ROOT / "hailo_model_zoo-*.whl")))
    with zipfile.ZipFile(c[-1]) as z:
        hit = [n for n in z.namelist() if n.endswith(suffix)]
        return z.read(hit[0]).decode()


def official_onnx() -> Path:
    cfg = zoo_read(f"cfg/networks/{ZOO_NET}.yaml")
    url = next(ln.split("url:", 1)[1].strip() for ln in cfg.split("\n") if ln.strip().startswith("url:"))
    d = OUT / "official"
    d.mkdir(parents=True, exist_ok=True)
    hit = list(d.rglob("*.onnx"))
    if hit:
        return hit[0]
    zp = d / Path(url).name
    print(f"[text] 공식 ONNX 다운로드: {url}", flush=True)
    urllib.request.urlretrieve(url, zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(d)
    return list(d.rglob("*.onnx"))[0]


def inspect(path: Path, start="Add_1", end="Add_1515") -> dict:
    import onnx
    m = onnx.load(str(path))
    g = m.graph
    inits = {i.name for i in g.initializer}
    consts = {o for n in g.node if n.op_type == "Constant" for o in n.output} | inits
    by_name = {n.name: n for n in g.node}
    prod = {o: n for n in g.node for o in n.output}

    def tinfo(vi):
        t = vi.type.tensor_type
        return {"name": vi.name, "elem_type": t.elem_type,
                "shape": [d.dim_value or d.dim_param for d in t.shape.dim]}

    res = {"inputs": [tinfo(i) for i in g.input if i.name not in inits],
           "outputs": [tinfo(o) for o in g.output], "n_nodes": len(g.node)}
    for key, nm in (("start", start), ("end", end)):
        n = by_name.get(nm)
        if n is None:   # 이름이 다르면 비슷한 이름을 기록
            res[key] = {"name": nm, "found": False,
                        "similar": [x for x in by_name if x.endswith(nm) or x.replace("/", "_").endswith(nm)][:5]}
            continue
        res[key] = {"name": nm, "found": True, "op": n.op_type, "inputs": list(n.input), "outputs": list(n.output),
                    "inputs_const": [i in consts for i in n.input],
                    "input_producers": [prod[i].op_type if i in prod else "graph_input/initializer" for i in n.input]}
    # 끝 노드 뒤에 남는 연산 (칩 밖에서 해야 할 일)
    if res["end"].get("found"):
        seen, frontier, after = set(), list(res["end"]["outputs"]), []
        while frontier:
            t = frontier.pop()
            for n in g.node:
                if t in n.input and n.name not in seen:
                    seen.add(n.name); after.append(n.op_type); frontier += list(n.output)
        res["ops_after_end"] = after
    return res


def ort_outputs(path: Path, tensors: list[str], feed_name: str, feeds: np.ndarray) -> list[np.ndarray]:
    """그래프 중간 텐서를 출력으로 열어 하나씩 실행한다 (배치 1 고정 모델)."""
    import onnx
    import onnxruntime as ort
    m = onnx.load(str(path))
    have = {o.name for o in m.graph.output}
    for t in tensors:
        if t not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL   # 중간 텐서가 사라지지 않게
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    outs = [[] for _ in tensors]
    for x in feeds:
        r = s.run(tensors, {feed_name: x[None]})
        for k, v in enumerate(r):
            outs[k].append(np.asarray(v).reshape(v.shape[-2], v.shape[-1]) if np.asarray(v).ndim >= 2 else v)
    return [np.stack(o) for o in outs]


def cos_rows(a, b):
    a = a.reshape(len(a), -1); b = b.reshape(len(b), -1)
    return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def post_numpy(x, eot, ln_w, ln_b, proj, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps) * ln_w + ln_b
    e = y[np.arange(len(y)), eot] @ proj
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def main() -> None:
    import open_clip
    import torch
    t00 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rec: dict = {"timestamp": C.now()}
    onnx_path = official_onnx()
    rec["onnx"] = {"path": str(onnx_path), "size_mb": C.size_mb(onnx_path), "sha256": C.sha256_file(onnx_path)}
    rec["inspect"] = info = inspect(onnx_path)
    print("[text] ONNX 입력:", info["inputs"], "\n[text] 시작:", info["start"], "\n[text] 끝:", info["end"],
          "\n[text] 끝 뒤 연산:", info.get("ops_after_end"), flush=True)
    LOG.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))

    # 캡션: 캘리브 = 제외 목록 이미지의 캡션, 간이 확인 = select 캡션 500개
    ann = json.loads(R.CAP_JSON.read_text())
    excl = set(R.read_split("exclude")) if R.split_path("exclude").exists() else \
        set(Path(C.SPLITS / "coco_ret_exclude.txt").read_text().split())
    fid = {im["id"]: im["file_name"] for im in ann["images"]}
    cal_caps = sorted(a["caption"].strip() for a in ann["annotations"] if fid[a["image_id"]] in excl)
    _, img_of, sel_caps = R.captions_for("select")
    pick = np.linspace(0, len(sel_caps) - 1, N_CHECK).astype(int)
    chk_caps = [sel_caps[i] for i in pick]
    rec["captions"] = {"calib_source": "splits/coco_ret_exclude.txt 이미지의 captions_val2017", "n_calib": len(cal_caps),
                       "check_source": f"select 캡션 {N_CHECK}개 (일정 간격, 간이 확인용)"}

    model = open_clip.create_model(C.MODEL_NAME, pretrained=C.PRETRAINED, force_quick_gelu=True,
                                   cache_dir=str(C.CACHE)).eval()
    tok = open_clip.get_tokenizer(C.MODEL_NAME)
    grab = {}
    h1 = model.token_embedding.register_forward_hook(lambda m, i, o: grab.__setitem__("tok", o.detach()))
    def _ln_hook(m, i, o):          # 반환값이 있으면 출력이 바뀌므로 None 을 돌려준다
        grab["pre_ln"] = i[0].detach(); grab["ln_out"] = o.detach()
    h2 = model.ln_final.register_forward_hook(_ln_hook)

    def run_clip(caps):
        toks, ref, te, tr, lnout = [], [], [], [], []
        with torch.no_grad():
            for i in range(0, len(caps), 128):
                t = tok(caps[i:i + 128])
                e = model.encode_text(t).float().numpy()
                pre, lo = grab["pre_ln"], grab["ln_out"]
                if pre.shape[0] == 77 and pre.shape[1] != 77:      # (L, N, D) 이면 배치 우선으로
                    pre, lo = pre.permute(1, 0, 2), lo.permute(1, 0, 2)
                toks.append(t.numpy()); ref.append(e / np.linalg.norm(e, axis=1, keepdims=True))
                te.append(grab["tok"].float().numpy()); tr.append(pre.float().numpy()); lnout.append(lo.float().numpy())
        return np.concatenate(toks), np.concatenate(ref), np.concatenate(te), np.concatenate(tr), np.concatenate(lnout)

    cal_tok, _, cal_te, cal_tr, _ = run_clip(cal_caps)
    chk_tok, chk_ref, chk_te, chk_tr, chk_lo = run_clip(chk_caps)
    h1.remove(); h2.remove()
    pos = model.positional_embedding.detach().float().numpy()
    ln_w = model.ln_final.weight.detach().float().numpy(); ln_b = model.ln_final.bias.detach().float().numpy()
    proj = model.text_projection.detach().float().numpy()
    if proj.shape != (512, 512):
        proj = proj.T
    chk_eot = chk_tok.argmax(-1); cal_eot = cal_tok.argmax(-1)

    # 4) 칩 밖 후처리 재현 확인 (open_clip 안에서 잡은 트랜스포머 출력 -> numpy 후처리 vs encode_text)
    c_post = cos_rows(post_numpy(chk_tr, chk_eot, ln_w, ln_b, proj), chk_ref)
    rec["host_post_vs_open_clip"] = {"min_cos": float(c_post.min()), "mean_cos": float(c_post.mean())}
    print(f"[text] numpy 후처리 vs open_clip encode_text: 최소 코사인 {c_post.min():.6f}", flush=True)

    # 끝 노드(Add_1515)의 상수 입력이 ln_final.bias 면 LayerNorm 까지 칩 안에서 한다 -> 비교 대상은 ln_final 출력
    ln_in_graph = any("ln_final" in i for i in info["end"].get("inputs", []))
    target = chk_lo if ln_in_graph else chk_tr
    rec["ln_final_in_graph"] = bool(ln_in_graph)
    print(f"[text] ln_final 이 그래프(칩) 안에 있음: {ln_in_graph}", flush=True)

    # 3) Hailo 입력 형식 결정: ONNX 를 그래프 입력부터 끝 노드까지 돌려 open_clip 트랜스포머 출력과 비교
    inp = info["inputs"][0]
    end_out = info["end"]["outputs"][0]
    start = info["start"]
    is_int = inp["elem_type"] in (6, 7)          # INT32 / INT64
    n_try = 16
    if is_int:
        # 토큰 id 를 넣고, 시작 노드의 상수가 아닌 입력(= Hailo 입력)과 끝 노드 출력을 꺼낸다
        hin = [t for t, c in zip(start["inputs"], start["inputs_const"]) if not c][0]
        dt = np.int64 if inp["elem_type"] == 7 else np.int32
        a_in, a_end = ort_outputs(onnx_path, [hin, end_out], inp["name"], chk_tok[:n_try].astype(dt))
        cand = {"tok": chk_te[:n_try], "tok_plus_pos": chk_te[:n_try] + pos}
        cmp = {k: float(cos_rows(v, a_in).min()) for k, v in cand.items()}
        choice = max(cmp, key=cmp.get)
        c_end = cos_rows(a_end, target[:n_try])
        rec["hailo_input"] = {"onnx_input": "token ids", "hailo_input_tensor": hin, "match_cos": cmp, "choice": choice,
                              "onnx_end_vs_open_clip_min_cos": float(c_end.min())}
    else:
        cand = {"tok": chk_te[:n_try], "tok_plus_pos": chk_te[:n_try] + pos}
        cmp = {}
        for k, v in cand.items():
            (a_end,) = ort_outputs(onnx_path, [end_out], inp["name"], v.astype(np.float32))
            cmp[k] = float(cos_rows(a_end, target[:n_try]).min())
        choice = max(cmp, key=cmp.get)
        rec["hailo_input"] = {"onnx_input": "float 77x512", "end_vs_open_clip_min_cos_by_candidate": cmp,
                              "choice": choice}
    print("[text] Hailo 입력 판정:", rec["hailo_input"], flush=True)
    ok = max(cmp.values()) > 0.999
    rec["hailo_input"]["verified"] = bool(ok)

    def make(te):
        return (te + pos if choice == "tok_plus_pos" else te).astype(np.float32)

    np.savez_compressed(OUT / "prep.npz", calib_in=make(cal_te), calib_eot=cal_eot,
                        check_in=make(chk_te), check_eot=chk_eot, check_ref=chk_ref,
                        check_img_of=img_of[pick], check_pick=pick, ln_w=ln_w, ln_b=ln_b, proj=proj,
                        ln_in_graph=np.array(ln_in_graph))
    rec["prep_npz"] = str(OUT / "prep.npz")
    rec["seconds"] = round(time.time() - t00, 1)
    LOG.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
    print(f"[text] 저장: {OUT / 'prep.npz'} (캘리브 {len(cal_caps)}개, 확인 {N_CHECK}개). 검증 {'통과' if ok else '실패'}")
    if not ok:
        raise SystemExit("!! Hailo 입력 형식을 open_clip 과 맞추지 못했습니다. logs/dfc/text_prep.json 확인")


if __name__ == "__main__":
    main()
