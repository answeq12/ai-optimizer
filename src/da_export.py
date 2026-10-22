"""E33: 선택된 적응 가중치로 FP32 ONNX 만들기 + P1 일치 검사. HALIO_MODELS_SUBDIR=models_da 로 실행.

  HALIO_MODELS_SUBDIR=models_da python -m src.da_export

- 원본 open_clip 모델에 da_tail_chosen.pt(블록 8~11, ln_post, proj)만 덮어써서 기존 export 함수 그대로 변환
- 텍스트 임베딩(CIFAR 프롬프트, COCO 캡션)은 텍스트 인코더를 얼렸으므로 원본 models/ 의 파일을 복사
- 원본 FP32 ONNX 와 연산 구성(연산 종류별 개수)이 같은지 확인 → 속도 조건 동일 근거
"""
from __future__ import annotations

import collections
import json
import shutil
import sys

import numpy as np
from PIL import Image

from . import common as C
from . import da_common as D
from .export_onnx import export, fold_constants, parity


def copy_text(orig):
    for f in ["text_emb_cifar100.npy", "cifar100_classes.json"] + sorted(p.name for p in orig.glob("coco_ret_text_*")):
        if (orig / f).exists():
            shutil.copy2(orig / f, C.MODELS / f)


def main():
    import onnx
    import torch
    assert C.MODELS.name == "models_da", "HALIO_MODELS_SUBDIR=models_da 로 실행하세요"
    C.ensure_dirs()
    if "--copy-text-only" in sys.argv:          # 텍스트 인코더는 얼렸으므로 원본 임베딩을 그대로 쓴다
        copy_text(C.STORE / "models"); return
    m = D.load_clip()
    tail = D.make_tail(m.visual)
    st = torch.load(D.DA_DIR / "da_tail_chosen.pt")
    tail.load_state_dict(st)
    m.eval()
    info = export(m, C.RAW_ONNX)
    info.update(fold_constants(C.RAW_ONNX, C.FP32_ONNX))
    data, _ = C.load_cifar100("train")
    val = C.read_index_list(C.SPLITS / "val.txt")[:20]
    par = parity(m, C.FP32_ONNX, np.stack([C.preprocess(Image.fromarray(data[i])) for i in val]))
    orig = C.STORE / "models"
    copy_text(orig)
    ops = lambda p: collections.Counter(n.op_type for n in onnx.load(str(p), load_external_data=False).graph.node)
    same = ops(C.FP32_ONNX) == ops(orig / C.FP32_ONNX.name)
    log = {"export": info, "parity": par, "same_op_counts_as_original": same,
           "alpha": json.loads((D.DA_DIR / "da_tail_chosen.json").read_text()), "timestamp": C.now()}
    (C.LOGS / "p1_parity_da.json").write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"[da-export] {C.FP32_ONNX}: {info['size_mb']} MB, 노드 {info['nodes']}개, 원본과 연산 구성 동일 {same}")
    print(f"[da-export] P1 최대 오차 {par['max_abs_err']:.2e}, 최소 코사인 {par['min_cosine']:.6f} -> {'통과' if par['passed'] else '실패'}")
    if not (par["passed"] and same):
        sys.exit(1)


if __name__ == "__main__":
    main()
