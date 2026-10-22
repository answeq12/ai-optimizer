"""연산자 커버리지 확인 + ORT 프로파일링 (P3.5).

  python -m src.coverage --models A0=models/clip_vitb16_image_fp32.onnx \
      A1=models/a1_dynamic.onnx A2=models/a2_static_minmax.onnx

이 단계가 없으면 동적과 정적의 비교가 무효다. 덜 양자화한 쪽이 더 정확한 것은 당연하기 때문이다.
세 가지를 따로 본다.
  1) 파일에 저장된 그래프의 연산 종류별 개수
  2) ORT 가 실행 직전에 최적화한 그래프의 연산 종류별 개수 (QDQ 가 실제로 정수 연산으로 합쳐졌는지)
  3) 프로파일의 연산 종류별 실행 시간 (정수 커널이 실제로 시간을 얼마나 차지하는지)

노드 개수와 실제 INT8 실행 비율은 다르다. QDQ 가 붙어 있어도 런타임에 합쳐지지 않으면 이득이 없다.

결과: results/coverage.json, results/coverage.md, logs/profile_<코드>.json,
      <모델 폴더>/runtime_optimized/optimized_<코드>.onnx
"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from . import common as C
from .backends import make_ort_session
from .benchmark import ensure_input

# 가중치를 가진 핵심 연산(선형층·합성곱)의 실수형과 정수형
FLOAT_COMPUTE = {"MatMul", "FusedMatMul", "Gemm", "FusedGemm", "Conv", "FusedConv"}
INT_COMPUTE = {"MatMulInteger", "ConvInteger", "QLinearMatMul", "QLinearConv", "QGemm",
               "DynamicQuantizeMatMul", "MatMulIntegerToFloat"}
PROFILE_RUNS = 30


def is_int_kernel(op: str) -> bool:
    return (op in INT_COMPUTE or op.startswith("QLinear") or op.endswith("Integer")
            or op.startswith("QAttention") or op == "QGemm")


def op_counts(model_path) -> dict[str, int]:
    import onnx
    m = onnx.load(str(model_path), load_external_data=False)
    return dict(collections.Counter(n.op_type for n in m.graph.node))


def has_weight_input(model_path) -> dict[str, dict[str, int]]:
    """MatMul 을 '가중치와의 곱'과 '활성값끼리의 곱'(어텐션)으로 나눠 센다."""
    import onnx
    m = onnx.load(str(model_path), load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    const = {o for n in m.graph.node if n.op_type in ("Constant", "DequantizeLinear")
             and all(i in inits for i in n.input) for o in n.output}
    res = collections.defaultdict(lambda: {"weight": 0, "activation": 0})
    for n in m.graph.node:
        if n.op_type in ("MatMul", "MatMulInteger", "QLinearMatMul"):
            w = any(i in inits or i in const for i in n.input)
            res[n.op_type]["weight" if w else "activation"] += 1
    return {k: dict(v) for k, v in res.items()}


def analyze(code: str, model_path: str, threads: int) -> dict:
    x = np.load(ensure_input())
    opt_dir = C.MODELS / "runtime_optimized"       # 수백 MB 라 모델 폴더 쪽에 둔다
    opt_dir.mkdir(parents=True, exist_ok=True)
    opt_path = opt_dir / f"optimized_{code}.onnx"
    prefix = str(C.LOGS / f"_prof_{code}")
    sess = make_ort_session(model_path, threads, profile_prefix=prefix, optimized_out=str(opt_path))
    name = sess.get_inputs()[0].name
    for _ in range(PROFILE_RUNS):
        sess.run(None, {name: x})
    prof_file = Path(sess.end_profiling())
    events = json.loads(prof_file.read_text())
    prof_dst = C.LOGS / f"profile_{code}.json"
    shutil.move(str(prof_file), prof_dst)

    t_by_op: dict[str, float] = collections.defaultdict(float)
    for e in events:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            t_by_op[e["args"].get("op_name", "?")] += e["dur"]
    total = sum(t_by_op.values()) or 1.0
    t_int = sum(t for op, t in t_by_op.items() if is_int_kernel(op))

    file_ops = op_counts(model_path)
    run_ops = op_counts(opt_path)
    n_int = sum(c for op, c in run_ops.items() if op in INT_COMPUTE or op.startswith("QLinearConv"))
    n_float = sum(c for op, c in run_ops.items() if op in FLOAT_COMPUTE)
    conv_int = sum(c for op, c in run_ops.items() if op in ("QLinearConv", "ConvInteger"))
    conv_float = run_ops.get("Conv", 0)
    return {
        "model_path": model_path, "file_ops": file_ops, "runtime_ops": run_ops,
        "runtime_matmul_split": has_weight_input(opt_path),
        "compute_nodes_int": n_int, "compute_nodes_float": n_float,
        "quantized_op_ratio": round(n_int / (n_int + n_float), 4) if (n_int + n_float) else 0.0,
        "int_kernel_time_share": round(t_int / total, 4),
        "patch_embed_conv_quantized": bool(conv_int and not conv_float),
        "qdq_left_at_runtime": run_ops.get("QuantizeLinear", 0) + run_ops.get("DequantizeLinear", 0),
        "top_ops_by_time": [
            {"op": op, "share": round(t / total, 4), "int_kernel": is_int_kernel(op)}
            for op, t in sorted(t_by_op.items(), key=lambda kv: -kv[1])[:12]],
        "profile": prof_dst.name, "profile_runs": PROFILE_RUNS, "threads": threads,
    }


def to_markdown(res: dict) -> str:
    codes = list(res)
    L = ["# 연산자 커버리지 (P3.5)", "",
         "정수 연산 비율 = 실행 직전 그래프에서 행렬곱·합성곱 연산(MatMul·Gemm·Conv 계열, 어텐션의 활성값끼리 곱 포함) 중 정수형인 것의 비율.",
         "정수 커널 시간 비중 = 프로파일에서 정수 커널이 차지한 실행 시간의 비율.", "",
         "| 항목 | " + " | ".join(codes) + " |", "|---|" + "---|" * len(codes)]

    def row(label, f):
        L.append(f"| {label} | " + " | ".join(str(f(res[c])) for c in codes) + " |")
    row("정수 연산 비율 (개수)", lambda r: f"{r['quantized_op_ratio']*100:.1f}%")
    row("정수 커널 시간 비중", lambda r: f"{r['int_kernel_time_share']*100:.1f}%")
    row("행렬곱·합성곱: 정수 / 실수", lambda r: f"{r['compute_nodes_int']} / {r['compute_nodes_float']}")
    row("patch embedding Conv 양자화", lambda r: "예" if r["patch_embed_conv_quantized"] else "아니오")
    row("실행 시 남은 Q/DQ 노드", lambda r: r["qdq_left_at_runtime"])
    L += ["", "## 실행 직전 그래프의 연산 종류별 개수", ""]
    ops = sorted({op for c in codes for op in res[c]["runtime_ops"]})
    L += ["| 연산 | " + " | ".join(codes) + " |", "|---|" + "---|" * len(codes)]
    for op in ops:
        L.append(f"| {op} | " + " | ".join(str(res[c]["runtime_ops"].get(op, 0)) for c in codes) + " |")
    L += ["", "## 실행 시간 상위 연산", ""]
    for c in codes:
        L.append(f"**{c}**: " + ", ".join(
            f"{t['op']} {t['share']*100:.1f}%{' (정수)' if t['int_kernel'] else ''}"
            for t in res[c]["top_ops_by_time"][:8]))
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="코드=경로")
    ap.add_argument("--threads", type=int, default=C.DEFAULT_THREADS)
    args = ap.parse_args()
    C.ensure_dirs()
    models = dict(m.split("=", 1) for m in args.models)
    res = {}
    for code, path in models.items():
        if not Path(path).exists():
            sys.exit(f"{code}: {path} 없음")
        res[code] = analyze(code, path, args.threads)
        r = res[code]
        print(f"[cov] {code}: 정수 연산 비율 {r['quantized_op_ratio']*100:.1f}%, "
              f"정수 커널 시간 {r['int_kernel_time_share']*100:.1f}%, "
              f"patch embed Conv 양자화 {'예' if r['patch_embed_conv_quantized'] else '아니오'}")
    jp = C.RESULTS / "coverage.json"
    prev = json.loads(jp.read_text()) if jp.exists() else {}
    prev.update(res)
    jp.write_text(json.dumps(prev, indent=2, ensure_ascii=False))
    (C.RESULTS / "coverage.md").write_text(to_markdown(prev))
    print("[cov] 저장:", C.RESULTS / "coverage.md")


if __name__ == "__main__":
    main()
