"""보고서 2장 '결과 요약' 칸 채우기: 파라미터 수, 연산량(MAC/GFLOPs), 파일 크기. 학습·측정 없음 (그래프 분석만)

  python -m src.model_stats            -> results/model_stats.md, results/model_stats.json

- 연산량 = MatMul/Gemm/Conv 의 곱셈-누산(MAC) 합. GFLOPs = 2 x GMAC 로 함께 적는다 (관례가 섞여 있어 둘 다)
  softmax·layernorm·GELU 같은 원소 단위 연산은 넣지 않는다 (전체의 1~2% 수준, 관례상 제외)
- 양자화는 파라미터 수와 MAC 수를 바꾸지 않는다. 바뀌는 것은 숫자 하나의 비트 수(파일 크기·메모리 대역폭)와 연산 종류(INT8 MAC)
- SPL(이상치 채널 분리)은 FP32 분할 그래프로 센다: 작은 행렬곱이 늘어난 만큼만 MAC 가 조금 늘어난다
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from . import common as C


def _dims(vi) -> list[int]:
    out = []
    for d in vi.type.tensor_type.shape.dim:
        out.append(d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1)   # 배치 등 동적 차원 = 1
    return out


def count(path: Path) -> dict:
    import onnx
    from onnx import shape_inference
    m = onnx.load(str(path))
    inits = {i.name: list(i.dims) for i in m.graph.initializer}
    params = int(sum(int(np.prod(d)) if d else 1 for d in inits.values()))
    g = shape_inference.infer_shapes(m).graph
    shp = {**{v.name: _dims(v) for v in list(g.value_info) + list(g.input) + list(g.output)}, **inits}
    macs, by_op, missing = 0, {}, 0
    for n in g.node:
        if n.op_type not in ("MatMul", "Gemm", "Conv"):
            continue
        a = shp.get(n.input[0]); b = shp.get(n.input[1]); o = shp.get(n.output[0])
        if a is None or b is None or o is None:
            missing += 1; continue
        if n.op_type == "MatMul":
            k = a[-1]; mac = int(np.prod(o)) * k
        elif n.op_type == "Gemm":
            ta = any(x.name == "transA" and x.i for x in n.attribute)
            k = a[0] if ta else a[-1]; mac = int(np.prod(o)) * k
        else:   # Conv: 출력 원소 x (Cin/groups x kh x kw)
            mac = int(np.prod(o)) * int(np.prod(b[1:]))
        macs += mac; by_op[n.op_type] = by_op.get(n.op_type, 0) + mac
    return {"file": str(path), "size_mb": path.stat().st_size / 1e6, "params_m": params / 1e6,
            "gmac": macs / 1e9, "gflops": 2 * macs / 1e9, "gmac_by_op": {k: v / 1e9 for k, v in by_op.items()},
            "nodes_shape_unknown": missing}


def main() -> None:
    M = C.MODELS
    items = [("A0 FP32 (baseline)", M / "clip_vitb16_image_fp32.onnx", True),
             ("SPL-b567-k32 FP32 분할 그래프 (연산량 계산용)", M / "imp" / "split_SPL-b567-k32.fp32.onnx", True),
             ("SPL-b567-k32 INT8 (최종, 파일 크기)", M / "imp" / "SPL-b567-k32.onnx", False),
             ("A1 동적 INT8 (파일 크기)", M / "a1_dynamic.onnx", False)]
    rows = []
    for name, p, do_count in items:
        if not p.exists():
            print(f"✘ 없음: {p}"); continue
        r = count(p) if do_count else {"file": str(p), "size_mb": p.stat().st_size / 1e6}
        r["name"] = name; rows.append(r); print(r)
    L = ["# 모델 통계 (파라미터 수·연산량·파일 크기)", "", f"생성 {C.now()}. 그래프 분석만, 측정 아님", "",
         "| 모델 | 파일 크기 (MB) | 파라미터 (M) | GMAC | GFLOPs (=2xGMAC) |", "|---|---|---|---|---|"]
    for r in rows:
        f = lambda k, fmt: (fmt.format(r[k]) if k in r else "—")
        L.append(f"| {r['name']} | {r['size_mb']:.1f} | {f('params_m','{:.2f}')} | {f('gmac','{:.2f}')} | {f('gflops','{:.2f}')} |")
    L += ["", "- 입력 1장(224x224), 배치 1. MatMul/Gemm/Conv 만 셈 (원소 단위 연산 제외)",
          "- 양자화는 파라미터 수와 MAC 수를 바꾸지 않는다. INT8 최종 모델의 파라미터·연산량은 FP32 분할 그래프 행과 같다",
          "- 파라미터 수는 ONNX initializer 원소 수 합 (상수 텐서 포함이라 공식 86M 과 소폭 다를 수 있음)"]
    (C.RESULTS / "model_stats.md").write_text("\n".join(L) + "\n")
    (C.RESULTS / "model_stats.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print("\n".join(L))


if __name__ == "__main__":
    main()
