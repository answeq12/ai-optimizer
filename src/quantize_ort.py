"""ONNX Runtime 양자화. 동적(A1)과 정적(A2, A3a, A3b)을 같은 진입점에서 만든다.

  # A1 동적 INT8
  python -m src.quantize_ort --mode dynamic --out models/a1_dynamic.onnx

  # A2 정적 INT8 기본 (MinMax, per-tensor, COCO 64장)
  python -m src.quantize_ort --mode static --calib-list splits/calib_coco_64_s0.txt \
      --out models/a2_static_minmax.onnx

  # 탐침 예: per-channel + Percentile, LayerNorm 은 양자화에서 제외
  python -m src.quantize_ort --mode static --calib-list splits/calib_coco_64_s0.txt \
      --per-channel --calib-method percentile --protect layernorm --out models/probe.onnx

"기본 설정을 썼다"는 재현 근거가 되지 않으므로, 실제로 넘긴 인자와 라이브러리 버전,
캘리브레이션 이미지 목록의 해시를 <out>.json 에 전부 남긴다.

  # A2m: 정적 INT8 을 A1 과 같은 대상(가중치 MatMul)으로 제한
  python -m src.quantize_ort --mode static --calib-list splits/calib_coco_64_s0.txt \
      --weight-matmul-only --out models/a2m_minmax.onnx

ORT 가 지원하는 범위 추정 방식은 MinMax / Entropy / Percentile 이다. MSE 는 없다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import common as C

PREP_ONNX = C.MODELS / "clip_vitb16_image_fp32.prep.onnx"

# 민감 레이어 보호 프리셋. Hailo 공식 설정이 16비트로 남긴 지점에서 가져온 "힌트"이며 검증된 정답이 아니다.
# 값은 (노드 op_type, 노드 이름 정규식). 둘 다 맞는 노드를 양자화 대상에서 뺀다.
PROTECT = {
    "layernorm": ("LayerNormalization", r".*"),
    "residual_add": ("Add", r"/transformer/resblocks\.\d+/Add(_1)?$"),
    "patch_embed": ("Conv", r".*"),
    "softmax": ("Softmax", r".*"),
    "final_proj": ("MatMul", r"^/MatMul$|/proj"),
}


def preprocess_model(force: bool = False) -> Path:
    """양자화 전 전처리(형상 추론). ORT 가 정적 양자화 전에 권장하는 단계.

    동적·정적 모두 이 파일을 입력으로 써서 출발점을 같게 한다.
    """
    if PREP_ONNX.exists() and not force:
        return PREP_ONNX
    from onnxruntime.quantization.shape_inference import quant_pre_process
    # 그래프 정리는 export 단계에서 이미 했다. 여기서 ORT 최적화를 다시 돌리면 MatMul+Add 가
    # Gemm 으로 합쳐지므로(export_onnx.fold_constants 주석 참조) 형상 추론만 한다.
    quant_pre_process(str(C.FP32_ONNX), str(PREP_ONNX), skip_optimization=True)
    how = "symbolic+onnx shape inference (ORT optimization skipped)"
    (PREP_ONNX.with_suffix(".json")).write_text(json.dumps(
        {"source": C.FP32_ONNX.name, "source_sha256": C.sha256_file(C.FP32_ONNX), "how": how,
         "timestamp": C.now()}, indent=2))
    print(f"[prep] {PREP_ONNX.name} 생성 ({how})")
    return PREP_ONNX


def nodes_matching(model_path: Path, op_type: str, pattern: str) -> list[str]:
    import onnx
    m = onnx.load(str(model_path), load_external_data=False)
    rx = re.compile(pattern)
    return [n.name for n in m.graph.node if n.op_type == op_type and rx.search(n.name)]


def weight_matmul_nodes(model_path: Path) -> list[str]:
    """가중치(상수)와 곱하는 MatMul/Gemm 만 고른다. 어텐션의 활성값끼리 곱(QK^T, AV)은 뺀다.

    동적 양자화(A1)가 실제로 양자화한 대상과 같은 집합이다(coverage.md 기준 A1: 정수 49 / 실수 25).
    정적 양자화를 같은 대상으로 제한해야 두 방식의 차이가 "범위 계산 방식" 하나로 좁혀진다.
    (전 연산 정적 양자화 A2 는 Softmax·Sigmoid·Mul·Add 까지 INT8 로 바꿔 top-1 2.09% 로 붕괴했다)
    """
    import onnx
    m = onnx.load(str(model_path), load_external_data=False)
    inits = {i.name for i in m.graph.initializer}
    const = {o for n in m.graph.node if n.op_type == "Constant" for o in n.output}
    return [n.name for n in m.graph.node
            if n.op_type in ("MatMul", "Gemm") and any(i in inits or i in const for i in n.input)]


HIST_CHUNK = 2


class _Chunk:
    """원래 reader 에서 최대 k 장만 꺼내 주는 얇은 래퍼."""

    def __init__(self, first, reader, k):
        self.first, self.reader, self.left = first, reader, k

    def get_next(self):
        if self.left <= 0:
            return None
        self.left -= 1
        if self.first is not None:
            x, self.first = self.first, None
            return x
        return self.reader.get_next()


def patch_histogram_chunking(k: int) -> None:
    """Percentile/Entropy 캘리브레이션의 메모리 폭증을 막는다.

    ORT 의 HistogramCalibrater.collect_data 는 모든 이미지의 모든 중간 텐서를 메모리에 모은 뒤에야
    히스토그램을 만든다(CalibMaxIntermediateOutputs 옵션은 MinMax 에만 적용된다). ViT-B/16 은
    이미지 한 장당 중간 텐서가 수백 MB 라 64장이면 수십 GB 가 된다. collect_data 를 k 장씩 나눠
    부르도록 감싼다. 히스토그램은 호출 사이에 누적된다.
    """
    from onnxruntime.quantization import calibrate as cal
    cls = getattr(cal, "HistogramCalibrater", None)
    if cls is None:
        print("[quant] 경고: HistogramCalibrater 를 찾지 못해 분할 집계를 적용하지 못했습니다. 메모리 사용에 주의.")
        return
    if getattr(cls, "_chunk_patched", False):
        return
    orig = cls.collect_data

    def collect_data(self, data_reader):
        got = False
        while True:
            first = data_reader.get_next()
            if not first:
                break
            orig(self, _Chunk(first, data_reader, k))
            got = True
        if not got:
            raise ValueError("No data is collected.")

    cls.collect_data = collect_data
    cls._chunk_patched = True


class CalibReader:
    """캘리브레이션 입력 공급자. 배치 1로 한 장씩 넘긴다."""

    def __init__(self, list_path: Path, input_name: str = "image"):
        self.data = C.load_calib_images(list_path)
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        x = self.data[self.i:self.i + 1]
        self.i += 1
        return {self.input_name: x}

    def rewind(self):
        self.i = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dynamic", "static"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib-list", default=None, help="정적 양자화용 캘리브레이션 목록 (splits/calib_*.txt)")
    ap.add_argument("--calib-method", default="minmax", choices=["minmax", "percentile", "entropy"])
    ap.add_argument("--percentile", type=float, default=99.9)
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--reduce-range", action="store_true")
    ap.add_argument("--weight-type", default="qint8", choices=["qint8", "quint8"])
    ap.add_argument("--activation-type", default="quint8", choices=["qint8", "quint8"])
    ap.add_argument("--op-types", default=None, help="쉼표 구분. 생략하면 ORT 기본 대상")
    ap.add_argument("--protect", default="", help=f"쉼표 구분 프리셋: {', '.join(PROTECT)}")
    ap.add_argument("--exclude-regex", default=None, help="이름이 정규식에 맞는 노드를 양자화에서 제외")
    ap.add_argument("--src", default=None, help="양자화할 FP32 모델 (기본: 전처리된 원본). 그래프를 고친 변형 모델용")
    ap.add_argument("--include-attn-matmul", action="store_true",
                    help="--weight-matmul-only 에 어텐션의 활성값끼리 곱(QK^T, AV) MatMul 도 포함")
    ap.add_argument("--quant-format", default="qdq", choices=["qdq", "qoperator"],
                    help="정적 양자화 그래프 형식. qdq = Q/DQ 노드 삽입(기본), qoperator = QLinearMatMul 등 정수 연산자로 직접 치환")
    ap.add_argument("--exclude-nodes-file", default=None,
                    help="한 줄에 노드 이름 하나. 적힌 노드를 양자화에서 제외 (src.sensitivity names 가 만든 파일)")
    ap.add_argument("--weight-matmul-only", action="store_true",
                    help="가중치와 곱하는 MatMul/Gemm 만 양자화 (A1 동적과 같은 대상). Softmax·GELU·Add·Conv 는 FP32")
    ap.add_argument("--dynamic-include-conv", action="store_true",
                    help="동적 양자화에서 patch embedding Conv 도 양자화 (ORT 버전에 따라 실행 불가)")
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                          quantize_dynamic, quantize_static)
    C.ensure_dirs()
    src = Path(args.src) if args.src else preprocess_model()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    qtype = {"qint8": QuantType.QInt8, "quint8": QuantType.QUInt8}
    op_types = [s.strip() for s in args.op_types.split(",")] if args.op_types else None

    exclude: list[str] = []
    protect = [p for p in args.protect.split(",") if p]
    for p in protect:
        if p not in PROTECT:
            sys.exit(f"알 수 없는 프리셋: {p}")
        exclude += nodes_matching(src, *PROTECT[p])
    if args.exclude_regex:
        m = onnx.load(str(src), load_external_data=False)
        rx = re.compile(args.exclude_regex)
        exclude += [n.name for n in m.graph.node if rx.search(n.name)]
    if args.exclude_nodes_file:
        listed = [ln.strip() for ln in Path(args.exclude_nodes_file).read_text().splitlines() if ln.strip()]
        known = {n.name for n in onnx.load(str(src), load_external_data=False).graph.node}
        missing = [n for n in listed if n not in known]
        if missing:
            sys.exit(f"제외 목록에 그래프에 없는 노드가 있습니다: {missing[:5]}")
        exclude += listed
    exclude = sorted(set(exclude))
    if (protect or args.exclude_regex or args.exclude_nodes_file) and not exclude:
        sys.exit("제외 조건에 맞는 노드가 하나도 없습니다. 노드 이름을 확인하세요.")

    include = None
    if args.weight_matmul_only:
        ex = set(exclude)
        include = [n for n in weight_matmul_nodes(src) if n not in ex]
        if args.include_attn_matmul:
            _m = onnx.load(str(src), load_external_data=False)
            _w = set(weight_matmul_nodes(src))
            attn = [n.name for n in _m.graph.node if n.op_type == "MatMul" and n.name not in _w and n.name not in ex]
            include += attn
            print(f"[quant] 어텐션 활성값 MatMul 추가: {len(attn)}개")
        if not include:
            sys.exit("가중치 MatMul 을 하나도 찾지 못했습니다. 그래프를 확인하세요.")
        op_types = sorted(set(op_types or []) | {"MatMul", "Gemm"})
        print(f"[quant] 가중치 MatMul 만 양자화: {len(include)}개 (A1 동적은 49개)")

    meta = {
        "quant_mode": args.mode, "source_model": src.name, "output_model": out.name,
        "weight_type": args.weight_type, "per_channel": args.per_channel,
        "reduce_range": args.reduce_range,
        "op_types_to_quantize": op_types or "ORT 기본값",
        "protect_presets": protect, "exclude_regex": args.exclude_regex,
        "exclude_nodes_file": args.exclude_nodes_file,
        "n_excluded_nodes": len(exclude), "excluded_nodes": exclude,
        "onnxruntime": ort.__version__, "onnx": onnx.__version__,
    }
    meta.update({"src_override": args.src, "include_attn_matmul": args.include_attn_matmul})
    meta.update({"weight_matmul_only": args.weight_matmul_only,
                 "op_types_to_quantize": op_types or "ORT 기본값",
                 "n_nodes_to_quantize": len(include) if include is not None else "전체(제외 목록 외)",
                 "nodes_to_quantize": include or []})

    if args.mode == "dynamic":
        if not args.dynamic_include_conv:
            # 동적 방식은 Conv 를 ConvInteger 로 바꾸는데, int8 가중치 ConvInteger 는 ORT 버전에 따라
            # 커널이 없다(1.23 에서 NOT_IMPLEMENTED, 1.30 에서는 동작). 제출 모델이 심사자 환경에서
            # 안 돌 수 있으므로 기본값은 Conv(patch embedding 1개)를 FP32 로 남긴다.
            conv = nodes_matching(src, "Conv", r".*")
            exclude = sorted(set(exclude) | set(conv))
            meta.update({"dynamic_conv_excluded": conv, "n_excluded_nodes": len(exclude),
                         "excluded_nodes": exclude})
        kw = dict(weight_type=qtype[args.weight_type], per_channel=args.per_channel,
                  reduce_range=args.reduce_range, nodes_to_exclude=exclude)
        if op_types:
            kw["op_types_to_quantize"] = op_types
        if include is not None:
            kw["nodes_to_quantize"] = include
        quantize_dynamic(str(src), str(out), **kw)
        meta.update({"calib_method": "", "calib_source": "", "calib_size": "", "calib_seed": ""})
    else:
        if not args.calib_list:
            sys.exit("정적 양자화에는 --calib-list 가 필요합니다.")
        clist = Path(args.calib_list)
        method = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
                  "entropy": CalibrationMethod.Entropy}[args.calib_method]
        extra = {}
        if args.calib_method == "percentile":
            extra["CalibPercentile"] = args.percentile
        if args.calib_method in ("percentile", "entropy"):
            patch_histogram_chunking(HIST_CHUNK)
        reader = CalibReader(clist)
        qfmt = {"qdq": QuantFormat.QDQ, "qoperator": QuantFormat.QOperator}[args.quant_format]
        kw = dict(quant_format=qfmt, per_channel=args.per_channel,
                  reduce_range=args.reduce_range, activation_type=qtype[args.activation_type],
                  weight_type=qtype[args.weight_type], nodes_to_exclude=exclude,
                  calibrate_method=method, extra_options=extra)
        if op_types:
            kw["op_types_to_quantize"] = op_types
        if include is not None:
            kw["nodes_to_quantize"] = include
        quantize_static(str(src), str(out), reader, **kw)
        seed = re.search(r"_s(\d+)\.txt$", clist.name)
        meta.update({
            "quant_format": args.quant_format.upper(), "activation_type": args.activation_type,
            "calib_method": args.calib_method + (f"({args.percentile})" if args.calib_method == "percentile" else ""),
            "calib_list": clist.name, "calib_list_sha256": C.sha256_file(clist),
            "calib_source": C.calib_source_of(clist), "calib_size": len(reader.data),
            "calib_seed": int(seed.group(1)) if seed else "", "extra_options": extra,
            "histogram_chunk": HIST_CHUNK if args.calib_method in ("percentile", "entropy") else "",
        })

    onnx.checker.check_model(str(out))
    meta.update({"size_mb": C.size_mb(out), "sha256": C.sha256_file(out), "timestamp": C.now()})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[quant] {out.name}: {meta['size_mb']} MB ({args.mode}"
          f"{', ' + meta['calib_method'] if args.mode == 'static' else ''}"
          f"{', per-channel' if args.per_channel else ''}"
          f"{', 제외 ' + str(len(exclude)) + '개' if exclude else ''})")


if __name__ == "__main__":
    main()
