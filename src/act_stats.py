"""가중치 행렬곱 49개의 '입력 활성값' 분포를 FP32 모델에서 직접 잰다.

목적: "fc2(블록 6·7)가 정적 INT8 에 민감한 이유는 입력에 극단적으로 큰 값(이상치)이 있어서
per-tensor MinMax 눈금이 넓게 잡히고, 대부분의 값이 몇 칸 안 되는 눈금에 몰리기 때문" 이라는 가설을
측정값으로 확인하거나 기각한다.

  python -m src.act_stats                                   # A2m 과 같은 캘리브레이션(COCO 64 s0)
  python -m src.act_stats --calib-list splits/calib_cifar_64_s0.txt --tag cifar

지표 (텐서마다, 캘리브레이션 이미지 전체에서)
  absmax        |x| 의 최댓값. MinMax 정적 양자화의 눈금 범위를 정하는 값
  p999          |x| 의 99.9 백분위수. "대부분의 값" 의 크기
  outlier_ratio absmax / p999. 클수록 이상치가 범위를 끌어올린다
  bulk_levels   p999 구간이 차지하는 INT8 눈금 칸 수 (= 255 * p999 / (max - min)). 작을수록 해상도가 나쁘다
  hot_channels  채널별 |x| 최댓값이 absmax 의 절반을 넘는 채널 수 (이상치가 소수 채널에 몰려 있는지)
결과: results/act_stats<tag>.md / .json. 분류는 results/sensitivity/groups.json 을 쓴다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import common as C
from .quantize_ort import preprocess_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-list", default=str(C.SPLITS / "calib_coco_64_s0.txt"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=C.DEFAULT_THREADS)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    groups = json.loads((C.RESULTS / "sensitivity" / "groups.json").read_text())["nodes"]
    src = preprocess_model()
    m = onnx.load(str(src))
    by_name = {n.name: n for n in m.graph.node}
    targets = []                                     # (그룹 행, 입력 텐서 이름)
    for r in groups:
        node = by_name[r["name"]]
        targets.append((r, node.input[0]))
    have = {o.name for o in m.graph.output}
    for _, t in targets:
        if t not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    tmp = C.MODELS / "act_stats_tmp.onnx"
    onnx.save(m, str(tmp), save_as_external_data=False)

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL   # 중간 텐서가 융합으로 사라지지 않게
    sess = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
    names = [t for _, t in targets]
    # splits/stats_*.txt 는 측정 전용 목록 (캘리브레이션에 쓰지 않음) -> 평가 사진과 겹쳐도 허용
    xs = C.load_calib_images(Path(args.calib_list), measure_only=Path(args.calib_list).name.startswith("stats_"))

    absmax = {t: 0.0 for t in names}
    vmin = {t: np.inf for t in names}
    vmax = {t: -np.inf for t in names}
    chmax = {t: None for t in names}
    samples = {t: [] for t in names}
    rng = np.random.default_rng(0)
    for i in range(len(xs)):
        outs = sess.run(names, {sess.get_inputs()[0].name: xs[i:i + 1]})
        for t, a in zip(names, outs):
            a = a.reshape(-1, a.shape[-1])                       # [토큰, 채널]
            ab = np.abs(a)
            absmax[t] = max(absmax[t], float(ab.max()))
            vmin[t] = min(vmin[t], float(a.min()))
            vmax[t] = max(vmax[t], float(a.max()))
            cm = ab.max(axis=0)
            chmax[t] = cm if chmax[t] is None else np.maximum(chmax[t], cm)
            flat = ab.ravel()
            samples[t].append(flat[rng.integers(0, flat.size, size=20000)])   # 백분위수용 표본
    rows = []
    for r, t in targets:
        s = np.concatenate(samples[t])
        p999 = float(np.percentile(s, 99.9))
        span = vmax[t] - vmin[t]
        rows.append({"name": r["name"], "kind": r["kind"], "block": r["block"], "input": t,
                     "absmax": absmax[t], "min": vmin[t], "max": vmax[t], "p999": p999,
                     "outlier_ratio": absmax[t] / p999 if p999 > 0 else None,
                     "bulk_levels": 255 * p999 / span if span > 0 else None,
                     "hot_channels": int((chmax[t] > 0.5 * absmax[t]).sum()),
                     "top_channel": int(np.argmax(chmax[t]))})
    tmp.unlink(missing_ok=True)

    sfx = f"_{args.tag}" if args.tag else ""
    (C.RESULTS / f"act_stats{sfx}.json").write_text(json.dumps(
        {"calib_list": Path(args.calib_list).name, "n_images": int(len(xs)), "rows": rows}, indent=1))
    L = [f"# 가중치 행렬곱 입력 활성값 분포 (FP32, {Path(args.calib_list).name}, {len(xs)}장)", "",
         "bulk_levels = 값의 99.9%가 쓰는 INT8 눈금 칸 수 (255칸 중, per-tensor MinMax 기준). 작을수록 해상도가 나쁘다.", ""]
    for kind in ("qkv", "out", "fc1", "fc2"):
        L += [f"## {kind}", "",
              "| 블록 | absmax | p99.9 | absmax / p99.9 | bulk_levels | 이상치 채널 수 | 최대 채널 |",
              "|---|---|---|---|---|---|---|"]
        for r in sorted([x for x in rows if x["kind"] == kind], key=lambda x: x["block"]):
            L.append(f"| {r['block']} | {r['absmax']:.1f} | {r['p999']:.2f} | {r['outlier_ratio']:.1f} | "
                     f"{r['bulk_levels']:.1f} | {r['hot_channels']} | {r['top_channel']} |")
        L.append("")
    pr = [x for x in rows if x["kind"] == "proj"][0]
    L += [f"proj: absmax {pr['absmax']:.1f}, p99.9 {pr['p999']:.2f}, bulk_levels {pr['bulk_levels']:.1f}", "",
          "- 측정값이다. 민감도(results/sensitivity.md)와 나란히 보고 해석한다. 이 표만으로 인과를 단정하지 않는다"]
    out = C.RESULTS / f"act_stats{sfx}.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"[act] 저장: {out}")


if __name__ == "__main__":
    main()
