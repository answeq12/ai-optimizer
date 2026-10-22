"""zero-shot top-1 정확도 평가. 모든 구성에 같은 코드를 쓴다.

  python -m src.evaluate --model models/clip_vitb16_image_fp32.onnx --code A0 --quant-mode fp32

- 평가셋 기본값은 검증셋(val). test 는 --eval-set test --i-know-final 을 둘 다 줘야 열린다.
- 이미지별 정오답을 results/preds/<run_id>.npz 로 저장한다(짝지은 비교용).
- 원본(FP32) 예측이 있으면 FP32 대비 예측 일치율과 임베딩 코사인도 함께 기록한다.
- 정확도는 결정론적이므로 반복하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from PIL import Image

from . import common as C
from .backends import OnnxBackend


def run_eval(backend, images: np.ndarray, labels: np.ndarray, text_emb: np.ndarray):
    preds = np.empty(len(images), dtype=np.int64)
    embs = np.empty((len(images), text_emb.shape[1]), dtype=np.float32)
    t0 = time.time()
    for i, im in enumerate(images):
        e = backend.encode_image(C.preprocess(Image.fromarray(im))[None])
        e = e / np.linalg.norm(e)
        embs[i] = e
        preds[i] = int(np.argmax(text_emb @ e))
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(images)}  ({time.time()-t0:.0f}s)", flush=True)
    return preds, embs


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """같은 이미지에서 두 구성의 정오답을 짝지어 비교. 정확 이항 검정(양측)."""
    from math import comb
    b = int(np.sum(correct_a & ~correct_b))     # a 만 맞힘
    c = int(np.sum(~correct_a & correct_b))     # b 만 맞힘
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p_value": 1.0}
    k = min(b, c)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": b, "b_only": c, "p_value": p}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--code", required=True, help="구성 코드 (A0, A1, A2, A3a, ...)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--quant-mode", default="", help="fp32 / dynamic / static")
    ap.add_argument("--eval-set", default="val", choices=["val", "test"])
    ap.add_argument("--i-know-final", action="store_true",
                    help="test 를 여는 것은 최종 구성 확정 후 한 번뿐임을 확인")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (코드 검증용)")
    ap.add_argument("--threads", type=int, default=C.DEFAULT_THREADS)
    ap.add_argument("--out-csv", default=str(C.RESULTS / "accuracy.csv"))
    ap.add_argument("--meta", default=None, help="양자화 설정 json (quantize_ort 가 만든 것)")
    args = ap.parse_args()

    if args.eval_set == "test" and not args.i_know_final:
        sys.exit("test 는 최종 구성 확정 후에만 엽니다. 맞다면 --i-know-final 을 함께 주세요.")
    C.ensure_dirs()
    run_id = args.run_id or f"{args.code}-{args.eval_set}"
    images, labels = C.load_eval_set(args.eval_set)
    if args.limit:
        images, labels = images[:args.limit], labels[:args.limit]
    text_emb = np.load(C.TEXT_EMB)

    backend = OnnxBackend(args.model, code=args.code, threads=args.threads)
    print(f"[eval] {run_id}: {args.model} / {args.eval_set} {len(images)}장")
    preds, embs = run_eval(backend, images, labels, text_emb)
    correct = preds == labels
    acc = float(correct.mean())
    se = float(np.sqrt(acc * (1 - acc) / len(labels)))

    pdir = C.RESULTS / "preds"
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / f"{run_id}.npz", preds=preds, labels=labels, embs=embs)

    extra = {}
    ref = pdir / f"A0-{args.eval_set}.npz"
    if ref.exists() and run_id != f"A0-{args.eval_set}":
        r = np.load(ref)
        if len(r["preds"]) == len(preds):
            extra["agree_with_fp32"] = float((r["preds"] == preds).mean())
            extra["mean_cosine_to_fp32"] = float(np.mean(np.sum(r["embs"] * embs, axis=1)))
            extra["min_cosine_to_fp32"] = float(np.min(np.sum(r["embs"] * embs, axis=1)))
            extra["mcnemar_vs_fp32"] = mcnemar(r["preds"] == r["labels"], correct)

    meta = json.loads(open(args.meta, encoding="utf-8").read()) if args.meta else {}
    env = C.dump_env()
    C.append_row(args.out_csv, {
        "run_id": run_id, "config_code": args.code, "backend": backend.name,
        "model_name": C.MODEL_NAME, "model_path": str(args.model),
        "quant_mode": args.quant_mode or meta.get("quant_mode", ""),
        "calib_method": meta.get("calib_method", ""), "per_channel": meta.get("per_channel", ""),
        "reduce_range": meta.get("reduce_range", ""),
        "excluded_nodes": meta.get("n_excluded_nodes", ""),
        "calib_source": meta.get("calib_source", ""), "calib_size": meta.get("calib_size", ""),
        "calib_seed": meta.get("calib_seed", ""),
        "eval_set": args.eval_set, "eval_n": len(labels), "top1_acc": round(acc, 6),
        "n_threads": args.threads, "model_size_mb": C.size_mb(args.model),
        "env_hash": env, "timestamp": C.now(), "repeat_idx": 0,
    })
    summary = {"run_id": run_id, "top1_acc": acc, "std_err": se, "n": len(labels), **extra}
    (C.RESULTS / "preds" / f"{run_id}.json").write_text(json.dumps(summary, indent=2))
    msg = f"[eval] {run_id}: top-1 {acc*100:.2f}% (표준오차 ±{se*100:.2f}%p, n={len(labels)})"
    if extra:
        msg += f", FP32 대비 일치율 {extra['agree_with_fp32']*100:.2f}%, 평균 코사인 {extra['mean_cosine_to_fp32']:.4f}"
    print(msg)


if __name__ == "__main__":
    main()
