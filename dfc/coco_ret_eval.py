"""Hailo DFC 에뮬레이터로 COCO 텍스트→이미지 검색 평가. ORT 경로(src.coco_retrieval)와 같은 이미지·캡션·지표.

  python -m dfc.coco_ret_eval --har <quantized.har> --code H-official --fp-code H-FP
입력은 uint8 0~255 NHWC (normalize_in_net). FP(양자화 전) 임베딩은 --fp-code 를 주면 같이 계산한다.
--fp-code 결과는 ORT A0 임베딩과 이미지별 코사인을 비교해(공간 정합성) 기록한다.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from PIL import Image

from src import common as C
from src import coco_retrieval as R
from dfc.smoke_test import embed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--har", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--fp-code", default="")
    ap.add_argument("--split", default="select", choices=["select"])
    ap.add_argument("--hw-arch", default="hailo10h")
    a = ap.parse_args()
    from hailo_sdk_client import ClientRunner, InferenceContext
    files, img_of, _ = R.captions_for(a.split)
    txt = np.load(R.text_path(a.split))
    t0 = time.time()
    x = np.stack([C.preprocess_uint8(Image.open(C.COCO_DIR / f)) for f in files]).astype(np.float32)
    print(f"[hailo-ret] 이미지 {len(x)}장 준비 ({time.time() - t0:.0f}s)", flush=True)
    runner = ClientRunner(har=a.har, hw_arch=a.hw_arch)
    extra = {"har": a.har, "backend": "hailo-dfc-emulator", "input": "uint8 0~255 NHWC"}
    if a.fp_code:
        t0 = time.time()
        fp = embed(runner, InferenceContext.SDK_FP_OPTIMIZED, x)
        res = R.save_and_report(a.fp_code, a.split, fp, img_of, txt, {**extra, "context": "SDK_FP_OPTIMIZED",
                                                                       "seconds": round(time.time() - t0)})
        ref = R.OUT / f"A0-{a.split}.npz"
        if ref.exists():
            e0 = np.load(ref)["emb"]
            cos = np.sum(e0 * fp, axis=1)
            chk = {"hailo_fp_vs_ort_A0_min_cos": float(cos.min()), "mean_cos": float(cos.mean()), "n": len(cos)}
            (R.OUT / f"{a.fp_code}-vs-A0.json").write_text(json.dumps(chk, indent=2))
            print(f"[hailo-ret] Hailo FP vs ORT A0 이미지 임베딩 코사인: 평균 {cos.mean():.5f}, 최소 {cos.min():.5f}")
    t0 = time.time()
    q = embed(runner, InferenceContext.SDK_QUANTIZED, x)
    R.save_and_report(a.code, a.split, q, img_of, txt, {**extra, "context": "SDK_QUANTIZED",
                                                        "seconds": round(time.time() - t0)})


if __name__ == "__main__":
    main()
