"""CLIP ViT-B/16 이미지 인코더 ONNX export + P1 게이트(PyTorch 와 출력 일치) + 텍스트 임베딩 사전 계산.

  python -m src.export_onnx

산출물
  models/clip_vitb16_image_fp32.raw.onnx  PyTorch 가 내보낸 그대로 (DFC 파싱 시 대조용)
  models/clip_vitb16_image_fp32.onnx   이미지 인코더만. 배치 1, 입력 [1,3,224,224] 고정, opset 17.
                                       raw 를 상수 폴딩한 것. Baseline(A0)이자 양자화의 출발점
  models/text_emb_cifar100.npy         CIFAR-100 100개 클래스 프롬프트 임베딩 [100,512], L2 정규화
  models/cifar100_classes.json         클래스 이름과 프롬프트
  logs/p1_parity.json                  출력 비교 로그

P1 판정 기준: 실제 이미지 20장에서 최대 절대 오차 <= 1e-4, 코사인 유사도 >= 0.9999.
실패하면 종료 코드 1. 정확도 수치로 판단하지 않는다.

텍스트 인코더는 NPU 에 올리지 않으므로 export 하지 않는다. 임베딩은 여기서 한 번만 계산하고
이후 전 실험에서 이 파일을 그대로 재사용한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

import numpy as np
from PIL import Image

from . import common as C

PARITY_N = 20
MAX_ABS_TOL = 1e-4
COS_TOL = 0.9999


def load_model(pretrained: str | None):
    import open_clip
    model = open_clip.create_model(C.MODEL_NAME, pretrained=pretrained, force_quick_gelu=True,
                                   cache_dir=str(C.CACHE))
    model.eval()
    return model, open_clip.get_tokenizer(C.MODEL_NAME)


def export(model, out_path) -> dict:
    import torch
    visual = model.visual
    dummy = torch.zeros(1, 3, C.IMAGE_SIZE, C.IMAGE_SIZE, dtype=torch.float32)
    kwargs = dict(input_names=["image"], output_names=["embedding"],
                  opset_version=C.OPSET, do_constant_folding=True)
    # nn.MultiheadAttention 의 eval 전용 fast path(aten::_native_multi_head_attention)는
    # ONNX 로 내보낼 수 없다. export 하는 동안만 끄고, 비교 기준이 되는 PyTorch 추론은
    # 기본 설정(fast path 켜짐)으로 돌려 P1 이 실제 원본 동작과의 일치를 확인하게 한다.
    mha = getattr(torch.backends, "mha", None)
    prev = mha.get_fastpath_enabled() if mha else None
    if mha:
        mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                torch.onnx.export(visual, (dummy,), str(out_path), dynamo=False, **kwargs)
            except TypeError:           # dynamo 인자가 없는 구버전 torch
                torch.onnx.export(visual, (dummy,), str(out_path), **kwargs)
            exporter = "torchscript"
    finally:
        if mha:
            mha.set_fastpath_enabled(prev)
    import onnx
    raw_nodes = len(onnx.load(str(out_path), load_external_data=False).graph.node)
    return {"exporter": exporter, "opset": C.OPSET, "torch": torch.__version__,
            "onnx": onnx.__version__, "raw_nodes": raw_nodes}


def fold_constants(raw_path, out_path) -> dict:
    """export 직후 그래프를 정리한다 (상수 폴딩, 중복 제거). 가중치와 연산 의미는 바뀌지 않는다.

    PyTorch 가 내보낸 그래프에는 입력 형상이 고정인데도 형상을 실행 중에 계산하는 연산
    (Shape, Gather, Cast, Sqrt, ...)이 수백 개 남는다. ORT 의 기본(BASIC) 수준 그래프 최적화로
    이를 상수로 접는다. BASIC 수준은 표준 ONNX 연산만 남기므로 다른 런타임에서도 읽을 수 있다.
    """
    import onnx
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(out_path)
    # MatMul+Add 를 Gemm 으로 합치는 변환은 끈다. 합쳐지면 ORT 동적 양자화가 선형층을
    # 양자화 대상으로 인식하지 못해(동적 방식은 Gemm 미지원) 동적과 정적의 비교가 불공정해진다.
    ort.InferenceSession(str(raw_path), so, providers=["CPUExecutionProvider"],
                         disabled_optimizers=["MatMulAddFusion"])
    m = onnx.load(str(out_path))
    onnx.checker.check_model(m)
    domains = sorted({n.domain for n in m.graph.node if n.domain not in ("", "ai.onnx")})
    if domains:
        raise RuntimeError(f"표준이 아닌 연산 도메인이 남았습니다: {domains}")
    return {"nodes": len(m.graph.node), "size_mb": C.size_mb(out_path),
            "sha256": C.sha256_file(out_path), "onnxruntime": ort.__version__,
            "graph_cleanup": "ORT_ENABLE_BASIC, MatMulAddFusion 비활성"}


def parity(model, onnx_path, images: np.ndarray) -> dict:
    import onnxruntime as ort
    import torch
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_abs, cos_min = 0.0, 1.0
    with torch.no_grad():
        for x in images:
            x = x[None]
            a = model.visual(torch.from_numpy(x)).numpy()[0]
            b = sess.run(None, {"image": x})[0][0]
            max_abs = max(max_abs, float(np.abs(a - b).max()))
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            cos_min = min(cos_min, cos)
    return {"n": len(images), "max_abs_err": max_abs, "min_cosine": cos_min,
            "tol_max_abs": MAX_ABS_TOL, "tol_cosine": COS_TOL,
            "passed": bool(max_abs <= MAX_ABS_TOL and cos_min >= COS_TOL)}


def text_embeddings(model, tokenizer, names: list[str]) -> np.ndarray:
    import torch
    prompts = [C.PROMPT_TEMPLATE.format(n) for n in names]
    with torch.no_grad():
        e = model.encode_text(tokenizer(prompts)).float().numpy()
    return (e / np.linalg.norm(e, axis=1, keepdims=True)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-init", action="store_true",
                    help="가중치를 받지 않고 무작위 초기화로 실행 (코드 검증 전용. 결과 수치는 무의미)")
    args = ap.parse_args()
    C.ensure_dirs()

    model, tokenizer = load_model(None if args.random_init else C.PRETRAINED)
    info = export(model, C.RAW_ONNX)
    info.update(fold_constants(C.RAW_ONNX, C.FP32_ONNX))
    print(f"[export] {C.FP32_ONNX.name}: {info['size_mb']} MB, opset {info['opset']}, "
          f"노드 {info['raw_nodes']} -> {info['nodes']}개 (상수 폴딩)")

    data, _ = C.load_cifar100("train")
    val = C.read_index_list(C.SPLITS / "val.txt")[:PARITY_N]
    imgs = np.stack([C.preprocess(Image.fromarray(data[i])) for i in val])
    par = parity(model, C.FP32_ONNX, imgs)
    print(f"[P1] 최대 절대 오차 {par['max_abs_err']:.2e} (기준 {MAX_ABS_TOL:.0e}), "
          f"최소 코사인 {par['min_cosine']:.6f} (기준 {COS_TOL}) -> "
          f"{'통과' if par['passed'] else '실패'}")

    names = C.cifar100_class_names()
    emb = text_embeddings(model, tokenizer, names)
    np.save(C.TEXT_EMB, emb)
    C.CLASS_NAMES.write_text(json.dumps(
        {"template": C.PROMPT_TEMPLATE, "classes": names}, indent=2, ensure_ascii=False))
    print(f"[text] {C.TEXT_EMB.name}: {emb.shape}")

    log = {"export": info, "parity": par, "random_init": args.random_init,
           "model": C.MODEL_NAME, "pretrained": None if args.random_init else C.PRETRAINED,
           "env": C.env_info(), "timestamp": C.now()}
    (C.LOGS / "p1_parity.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    if not par["passed"]:
        sys.exit("[P1] 실패: opset, 전처리 순서, export 옵션을 점검하세요. 통과 전에는 다음 단계로 가지 않습니다.")


if __name__ == "__main__":
    main()
