"""최종 판정 (한 번만): Hailo 에뮬레이터로 COCO-confirm 검색 + CIFAR-100 test 10k 분류.

  python -m dfc.final_eval --har <quantized.har> --code H-splf-b7-k32 [--fp-code H-FP] --i-know-final
구성 동결(10/17) 뒤에만 연다. select 평가(dfc.coco_ret_eval)와 같은 전처리(uint8 NHWC)·지표.
결과: results/coco_ret/<code>-confirm.{json,npz}, results/preds/<code>-test.npz, results/final_hailo_<code>.json
"""
from __future__ import annotations

import argparse
import json
import sys
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
    ap.add_argument("--i-know-final", action="store_true")
    ap.add_argument("--hw-arch", default="hailo10h")
    ap.add_argument("--skip-coco", action="store_true")
    ap.add_argument("--skip-cifar", action="store_true")
    a = ap.parse_args()
    if not a.i_know_final:
        sys.exit("confirm·test 는 구성 동결 후 한 번만 엽니다. --i-know-final 을 함께 주세요.")
    from hailo_sdk_client import ClientRunner, InferenceContext
    runner = ClientRunner(har=a.har, hw_arch=a.hw_arch)
    extra = {"har": a.har, "backend": "hailo-dfc-emulator", "input": "uint8 0~255 NHWC"}
    out = {"code": a.code, "fp_code": a.fp_code, "har": a.har, "timestamp": C.now()}
    kinds = [(a.code, InferenceContext.SDK_QUANTIZED)] + ([(a.fp_code, InferenceContext.SDK_FP_OPTIMIZED)] if a.fp_code else [])
    if not a.skip_coco:
        files, img_of, _ = R.captions_for("confirm")
        txt = np.load(R.text_path("confirm"))
        x = np.stack([C.preprocess_uint8(Image.open(C.COCO_DIR / f)) for f in files]).astype(np.float32)
        print(f"[final] COCO-confirm 이미지 {len(x)}장 / 캡션 {len(txt)}개", flush=True)
        for code, ctx in kinds:
            t0 = time.time(); e = embed(runner, ctx, x)
            res = R.save_and_report(code, "confirm", e, img_of, txt, {**extra, "context": str(ctx), "seconds": round(time.time() - t0)})
            out[f"{code}_coco_confirm"] = {k: res[k] for k in ("R@1", "R@5", "R@10")}
    if not a.skip_cifar:
        data, labels = C.load_eval_set("test")
        t = np.load(C.TEXT_EMB)
        x = np.stack([C.preprocess_uint8(Image.fromarray(d)) for d in data]).astype(np.float32)
        print(f"[final] CIFAR-100 test {len(x)}장", flush=True)
        pdir = C.RESULTS / "preds"; pdir.mkdir(parents=True, exist_ok=True)
        for code, ctx in kinds:
            t0 = time.time(); e = embed(runner, ctx, x)
            p = np.argmax(e @ t.T, axis=1); acc = float((p == labels).mean())
            np.savez_compressed(pdir / f"{code}-test.npz", preds=p, labels=labels, embs=e)
            out[f"{code}_cifar_test_top1"] = acc
            print(f"[final] {code} CIFAR test top-1 {acc:.2%} ({time.time() - t0:.0f}s)", flush=True)
    (C.RESULTS / f"final_hailo_{a.code}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
