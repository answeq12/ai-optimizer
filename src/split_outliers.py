"""이상치 채널 분리. fc2 입력의 극단값이 소수(3~4개) 채널에 몰려 있다는 측정(act_stats)을 이용한다.

fc2: y = x W  (x: [1,197,3072], W: [3072,768])
  -> y = x[:, :, hot] W[hot, :]  +  x[:, :, rest] W[rest, :]
수학적으로 같은 FP32 모델을 만든다. 이후 정적 INT8 에서 hot 쪽 작은 행렬곱(k 채널)만 FP32 로 두고
rest 쪽(나머지 3072-k 채널)은 INT8 로 양자화하면, 눈금이 이상치에 끌려가지 않는다.
(층 전체를 FP32 로 두던 혼합 정밀도보다 FP32 계산량이 3072/k 배 작다)

  python -m src.split_outliers --blocks 6,7 --k 4 --out models/split_b67_k4.fp32.onnx
hot 채널은 캘리브레이션 이미지(기본 COCO 64 s0, 양자화와 같은 것)에서 채널별 |x| 최댓값 상위 k 개.
검증셋은 쓰지 않는다. 원본과 출력이 같은지(코사인 > 0.99999) 직접 확인하고, 아니면 멈춘다.
산출: <out>, <out>.json(선택 채널 기록), <out>.hot_nodes.txt(FP32 로 둘 노드 이름, quantize_ort --exclude-nodes-file 용)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import common as C
from .quantize_ort import preprocess_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", required=True, help="쉼표 구분 블록 번호 (fc2 대상)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--calib-list", default=str(C.SPLITS / "calib_coco_64_s0.txt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper, TensorProto

    blocks = [int(b) for b in args.blocks.split(",")]
    groups = json.loads((C.RESULTS / "sensitivity" / "groups.json").read_text())["nodes"]
    targets = {r["block"]: r["name"] for r in groups if r["kind"] == "fc2" and r["block"] in blocks}
    if len(targets) != len(blocks):
        sys.exit(f"fc2 노드를 찾지 못한 블록이 있습니다: {blocks} -> {targets}")

    src = preprocess_model()
    m = onnx.load(str(src))
    nodes = {n.name: n for n in m.graph.node}
    inits = {i.name: i for i in m.graph.initializer}
    xs = C.load_calib_images(Path(args.calib_list))

    # 1) 캘리브레이션 이미지로 채널별 |x| 최댓값
    probe = onnx.load(str(src))
    for b, name in targets.items():
        probe.graph.output.append(helper.make_tensor_value_info(nodes[name].input[0], TensorProto.FLOAT, None))
    tmp = C.MODELS / "split_probe_tmp.onnx"
    onnx.save(probe, str(tmp))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
    tnames = [nodes[targets[b]].input[0] for b in blocks]
    chmax = {t: None for t in tnames}
    for i in range(len(xs)):
        for t, a in zip(tnames, sess.run(tnames, {sess.get_inputs()[0].name: xs[i:i + 1]})):
            c = np.abs(a.reshape(-1, a.shape[-1])).max(axis=0)
            chmax[t] = c if chmax[t] is None else np.maximum(chmax[t], c)
    tmp.unlink(missing_ok=True)

    # 2) 그래프 고치기
    rec, hot_nodes = [], []
    new_nodes = []
    for n in m.graph.node:
        blk = next((b for b, nm in targets.items() if nm == n.name), None)
        if blk is None:
            new_nodes.append(n)
            continue
        x, wname = n.input[0], n.input[1]
        W = numpy_helper.to_array(inits[wname])
        cm = chmax[x]
        hot = np.sort(np.argsort(-cm)[:args.k]).astype(np.int64)
        rest = np.setdiff1d(np.arange(W.shape[0]), hot).astype(np.int64)
        p = f"/split/b{blk}"
        m.graph.initializer.extend([
            numpy_helper.from_array(hot, f"{p}/idx_hot"), numpy_helper.from_array(rest, f"{p}/idx_rest"),
            numpy_helper.from_array(W[hot, :].astype(np.float32), f"{p}/W_hot"),
            numpy_helper.from_array(W[rest, :].astype(np.float32), f"{p}/W_rest")])
        new_nodes += [
            helper.make_node("Gather", [x, f"{p}/idx_hot"], [f"{p}/x_hot"], name=f"{p}/Gather_hot", axis=-1),
            helper.make_node("Gather", [x, f"{p}/idx_rest"], [f"{p}/x_rest"], name=f"{p}/Gather_rest", axis=-1),
            helper.make_node("MatMul", [f"{p}/x_hot", f"{p}/W_hot"], [f"{p}/y_hot"], name=f"{p}/MatMul_hot"),
            helper.make_node("MatMul", [f"{p}/x_rest", f"{p}/W_rest"], [f"{p}/y_rest"], name=f"{p}/MatMul_rest"),
            helper.make_node("Add", [f"{p}/y_hot", f"{p}/y_rest"], list(n.output), name=f"{p}/Add_merge")]
        hot_nodes.append(f"{p}/MatMul_hot")
        bulk = float(np.percentile(cm[rest], 99.9)) if len(rest) else 0.0
        rec.append({"block": blk, "node": n.name, "hot_channels": hot.tolist(),
                    "hot_absmax": [float(cm[h]) for h in hot], "rest_absmax": float(cm[rest].max()),
                    "absmax_before": float(cm.max())})
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    used = {i for nd in m.graph.node for i in nd.input}
    keep = [i for i in m.graph.initializer if i.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    onnx.checker.check_model(m)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(out))

    # 3) 원본과 출력이 같은지 확인
    s0 = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    cos = []
    for i in range(min(8, len(xs))):
        a = s0.run(None, {s0.get_inputs()[0].name: xs[i:i + 1]})[0].ravel()
        b = s1.run(None, {s1.get_inputs()[0].name: xs[i:i + 1]})[0].ravel()
        cos.append(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
    if min(cos) < 0.99999:
        sys.exit(f"분리 모델 출력이 원본과 다릅니다 (최소 코사인 {min(cos):.6f}). 중단")
    meta = {"blocks": blocks, "k": args.k, "calib_list": Path(args.calib_list).name, "layers": rec,
            "equivalence_min_cosine": min(cos), "size_mb": C.size_mb(out), "timestamp": C.now()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    Path(str(out) + ".hot_nodes.txt").write_text("\n".join(hot_nodes) + "\n")
    for r in rec:
        print(f"[split] 블록 {r['block']}: hot {r['hot_channels']} (|x| {['%.1f' % v for v in r['hot_absmax']]}) "
              f"-> 나머지 최대 {r['rest_absmax']:.1f} (분리 전 {r['absmax_before']:.1f})")
    print(f"[split] 저장 {out.name}, 원본 대비 최소 코사인 {min(cos):.7f}")


if __name__ == "__main__":
    main()
