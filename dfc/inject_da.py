"""본선 준비: 적응 가중치(DA5 등)를 Hailo Model Zoo 공식 ONNX 에 덮어써 '공식 형식 + 적응 가중치' ONNX 를 만든다.

  python -m dfc.inject_da --models-subdir models_da5 --tag da5      # → <STORE>/hailo/official_da5.onnx

방법: 공식 ONNX 의 initializer 를 우리 원본 FP32 ONNX(models/clip_vitb16_image_fp32.onnx)의 initializer 와 '값'으로 맞춘다
(같은 값이면 같은 가중치). 맞은 것 중 적응 ONNX(같은 이름)에서 값이 바뀐 것만 공식 쪽에 덮어쓴다(전치 형태도 확인).
검사: ① 적응으로 바뀐 텐서가 전부 공식 쪽에서 찾아졌는지 ② 공식+적응 ONNX 와 우리 적응 FP32 ONNX 출력이 같은지(코사인 ≥ 0.9999, 캘리브 64장)
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from src import common as C


def _key(a):
    return hashlib.sha1(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest() + str(a.shape)


def main():
    import onnx, onnxruntime as ort
    from onnx import numpy_helper
    ap = argparse.ArgumentParser(); ap.add_argument("--models-subdir", default="models_da5"); ap.add_argument("--tag", default="da5")
    a = ap.parse_args()
    off_dir = C.STORE / "hailo" / "official"; off = next(off_dir.rglob("*.onnx"))
    ours0 = onnx.load(str(C.STORE / "models" / "clip_vitb16_image_fp32.onnx"))
    oursD = onnx.load(str(C.STORE / a.models_subdir / "clip_vitb16_image_fp32.onnx"))
    m = onnx.load(str(off))
    v0 = {i.name: numpy_helper.to_array(i) for i in ours0.graph.initializer}
    vD = {i.name: numpy_helper.to_array(i) for i in oursD.graph.initializer}
    changed = {n for n in v0 if n in vD and v0[n].shape == vD[n].shape and not np.array_equal(v0[n], vD[n])}
    lut = {}
    for n in changed:
        lut[_key(v0[n])] = (n, False)
        if v0[n].ndim == 2: lut[_key(v0[n].T)] = (n, True)
    hit, used = 0, set()
    for init in m.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype != np.float32: continue
        k = _key(arr)
        if k in lut:
            n, tr = lut[k]; new = vD[n].T if tr else vD[n]
            init.CopyFrom(numpy_helper.from_array(new.astype(np.float32), init.name)); hit += 1; used.add(n)
    miss = sorted(changed - used)
    print(f"[inject] 공식 ONNX: {off.name} / 적응으로 바뀐 텐서 {len(changed)}개 중 공식 쪽에서 찾아 바꾼 것 {len(used)}개 (덮어쓴 initializer {hit}개)")
    if miss: print("[inject] 못 찾은 텐서:", miss[:20])
    dst = C.STORE / "hailo" / f"official_{a.tag}.onnx"; onnx.save(m, str(dst))
    # 출력 비교 (공식 ONNX 입력 = 정규화된 NCHW, split_hailo 와 같은 규약)
    xs = C.load_calib_images(C.SPLITS / "calib_coco_64_s0.txt")
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s1 = ort.InferenceSession(str(dst), so, providers=["CPUExecutionProvider"]); s2 = ort.InferenceSession(str(C.STORE / a.models_subdir / "clip_vitb16_image_fp32.onnx"), so, providers=["CPUExecutionProvider"])
    s0 = ort.InferenceSession(str(off), so, providers=["CPUExecutionProvider"])
    def run(s, x): o = s.run(None, {s.get_inputs()[0].name: x})[0].reshape(len(x), -1); return o / np.linalg.norm(o, axis=1, keepdims=True)
    e1 = np.concatenate([run(s1, xs[i:i + 1]) for i in range(len(xs))]); e2 = np.concatenate([run(s2, xs[i:i + 1]) for i in range(len(xs))])
    e0 = np.concatenate([run(s0, xs[i:i + 1]) for i in range(len(xs))])
    cos, cos0 = float((e1 * e2).sum(1).min()), float((e0 * e2).sum(1).min())
    ok = (not miss) and cos >= 0.9999
    rec = {"official": str(off), "dst": str(dst), "changed": len(changed), "replaced": len(used), "missing": miss,
           "min_cos_vs_ours_adapted": cos, "min_cos_official_unmodified_vs_ours_adapted": cos0, "pass": ok}
    (C.LOGS / "dfc" / f"inject_{a.tag}.json").write_text(json.dumps(rec, indent=1))
    print(f"[inject] 공식+적응 vs 우리 적응 FP32: 최소 코사인 {cos:.6f} (덮어쓰기 전 공식 {cos0:.4f}) → {'통과' if ok else '실패'}")
    if not ok: sys.exit(1)


if __name__ == "__main__":
    main()
