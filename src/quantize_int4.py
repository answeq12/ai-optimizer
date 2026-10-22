"""가중치 전용 INT4 양자화 (W4A32). ORT 의 MatMulNBits 블록 양자화를 쓴다.

활성값은 FP32 로 두고 가중치만 4비트(블록마다 스케일)로 줄인다. 파일 크기를 크게 줄이는 손잡이이며,
Hailo-10H 가 INT4 를 지원한다는 공지 6장과 연결되는 실험이다.

  python -m src.quantize_int4 --out models/i4_all.onnx
  python -m src.quantize_int4 --out models/i4_ex67.onnx --exclude-nodes-file results/sensitivity/S-c67-fc2.nodes.txt
  python -m src.quantize_int4 --out models/i4_b32.onnx --block-size 32

대상은 MatMul 뿐이다(out_proj 는 그래프에서 Gemm 이라 FP32 로 남는다. 실제 대상 수를 메타에 기록한다).
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from . import common as C
from .quantize_ort import preprocess_model, weight_matmul_nodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--symmetric", action="store_true", default=True)
    ap.add_argument("--accuracy-level", type=int, default=None,
                    help="MatMulNBits 연산 정밀도 (None=FP32 계산, 4=INT8 계산). 정확도 실험은 None")
    ap.add_argument("--exclude-nodes-file", default=None)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    try:
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer as Q
    except ImportError:
        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer as Q

    src = preprocess_model()
    exclude = []
    if args.exclude_nodes_file:
        exclude = [ln.strip() for ln in Path(args.exclude_nodes_file).read_text().splitlines() if ln.strip()]
    m = onnx.load(str(src))
    wm = set(weight_matmul_nodes(src))
    targets = [n.name for n in m.graph.node if n.op_type == "MatMul" and n.name in wm and n.name not in exclude]

    sig = inspect.signature(Q.__init__).parameters
    kw = {"block_size": args.block_size, "is_symmetric": args.symmetric, "nodes_to_exclude": exclude}
    if "accuracy_level" in sig:
        kw["accuracy_level"] = args.accuracy_level
    q = Q(m, **{k: v for k, v in kw.items() if k in sig})
    q.process()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    q.model.save_model_to_file(str(out), use_external_data_format=False)

    qm = onnx.load(str(out), load_external_data=False)
    n_nbits = sum(1 for n in qm.graph.node if n.op_type == "MatMulNBits")
    meta = {"quant_mode": "weight_only_int4", "source_model": src.name, "output_model": out.name,
            "block_size": args.block_size, "symmetric": args.symmetric, "accuracy_level": args.accuracy_level,
            "n_excluded_nodes": len(exclude), "excluded_nodes": exclude,
            "n_matmul_targets_expected": len(targets), "n_matmulnbits_in_output": n_nbits,
            "gemm_left_fp32": sum(1 for n in qm.graph.node if n.op_type == "Gemm"),
            "onnxruntime": ort.__version__, "onnx": onnx.__version__,
            "size_mb": C.size_mb(out), "sha256": C.sha256_file(out), "timestamp": C.now()}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[int4] {out.name}: {meta['size_mb']} MB, MatMulNBits {n_nbits}개 (기대 {len(targets)}), "
          f"block {args.block_size}, 제외 {len(exclude)}")


if __name__ == "__main__":
    main()
