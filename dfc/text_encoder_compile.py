"""Hailo-10H 용 CLIP ViT-B/16 텍스트 인코더 HEF 컴파일 (DFC 환경 ~/venvs/hailo).

  python -m dfc.text_encoder_compile            # parse -> optimize -> 간이 평가 -> compile
먼저 파일럿 환경에서 `python -m src.text_hailo_prep` 을 실행해 prep.npz 를 만들어야 한다.

- ONNX·시작/끝 노드·모델 스크립트는 Hailo Model Zoo 5.4.0 공식 설정(clip_vit_b_16_text_encoder) 그대로
  (공식 alls: 대부분 층 a16_w16, finetune 켜짐). finetune 단계에서 실패하면 finetune 만 끄고 한 번 더 시도하고 기록한다
- 칩에 올라가는 범위: 트랜스포머 본체(입력 77x512 -> 출력 77x512). 토큰화·임베딩·ln_final·EOT 선택·투영은 칩 밖(CPU)
- 캘리브레이션: COCO 검색 '제외' 목록 이미지의 캡션 (평가 select/confirm 과 겹치지 않음)
- 간이 평가: select 캡션 500개. FP32(open_clip) 대비 코사인, 그리고 캡션 단위 R@1(갤러리 = select 이미지 2,405장)
  캡션 단위라 이미지 단위 공식 지표와 다르다. 결론 수치가 아니라 '쓸 만한가' 확인용
결과: logs/dfc/text_<시각>.json, <STORE>/hailo/clip_vit_b_16_text_encoder.{parsed,quantized,compiled}.har, .hailo10h.hef
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from src import common as C
from dfc.smoke_test import zoo_read, zoo_wheel, hn_layers

NET = "clip_vit_b_16_text_encoder"
HAILO_DIR = C.STORE / "hailo"
TEXT_DIR = HAILO_DIR / "text"
LOG_DIR = C.LOGS / "dfc"


def post_numpy(x, eot, ln_w, ln_b, proj, eps=1e-5, ln_in_graph=True):
    """칩 출력 -> (ln_final, 그래프에 없을 때만) -> EOT 위치 선택 -> text_projection -> 정규화"""
    if ln_in_graph:
        y = x
    else:
        mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
        y = (x - mu) / np.sqrt(var + eps) * ln_w + ln_b
    e = y[np.arange(len(y)), eot] @ proj
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw-arch", default="hailo10h")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-finetune", action="store_true", help="처음부터 finetune 을 끈다")
    ap.add_argument("--skip-compile", action="store_true")
    a = ap.parse_args()
    SFX = f".{a.tag}" if a.tag else ""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"text_{stamp}{SFX}.json"
    rec: dict = {"timestamp": C.now(), "net": NET, "hw_arch": a.hw_arch, "args": vars(a), "stages": {}}

    def save():
        log_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))

    def stage(name, fn, fatal=True):
        print(f"[text-dfc] ▶ {name}", flush=True)
        t0 = time.time()
        try:
            out = fn()
            rec["stages"][name] = {"ok": True, "seconds": round(time.time() - t0, 1)}
            print(f"[text-dfc] ✔ {name} ({rec['stages'][name]['seconds']}s)", flush=True)
            save()
            return out
        except Exception as e:
            rec["stages"][name] = {"ok": False, "seconds": round(time.time() - t0, 1),
                                   "error": f"{type(e).__name__}: {e}"[:3000],
                                   "traceback": traceback.format_exc()[-6000:]}
            save()
            print(f"[text-dfc] ✘ {name} 실패: {type(e).__name__}: {str(e)[:800]}", flush=True)
            if fatal:
                rec["failed_stage"] = name; save()
                print(f"[text-dfc] 로그: {log_path}")
                sys.exit(1)
            return None

    from hailo_sdk_client import ClientRunner, InferenceContext
    prep_info = json.loads((LOG_DIR / "text_prep.json").read_text())
    P = np.load(TEXT_DIR / "prep.npz")
    onnx_path = Path(prep_info["onnx"]["path"])
    start = prep_info["inspect"]["start"]["name"]; end = prep_info["inspect"]["end"]["name"]
    rec["onnx"] = prep_info["onnx"]; rec["nodes"] = {"start": start, "end": end}
    rec["hailo_input"] = prep_info.get("hailo_input")

    def do_parse():
        r = ClientRunner(hw_arch=a.hw_arch)
        r.translate_onnx_model(str(onnx_path), NET, start_node_names=[start], end_node_names=[end])
        r.save_har(str(HAILO_DIR / f"{NET}{SFX}.parsed.har"))
        return r
    runner = stage("parse", do_parse)

    def input_shape():
        for name, L in hn_layers(runner).items():
            if L.get("type") == "input_layer":
                s = L.get("output_shapes", [[None]])[0]
                return name, [int(v) if v not in (None, -1) else -1 for v in s]
        return None, None
    in_name, in_shape = input_shape()
    rec["hn_input"] = {"name": in_name, "shape": in_shape}
    print(f"[text-dfc] HN 입력 {in_name} {in_shape}", flush=True)

    def shape_for(x):
        if in_shape and len(in_shape) == 4:
            return x.reshape(len(x), *[d for d in in_shape[1:]]).astype(np.float32)
        return x.astype(np.float32)

    base_alls = zoo_read(f"cfg/alls/generic/{NET}.alls")
    rec["alls_source"] = f"{zoo_wheel().name}: cfg/alls/generic/{NET}.alls"
    no_ft = "post_quantization_optimization(finetune, policy=disabled)"

    def alls_without_finetune(s):
        return "\n".join(ln for ln in s.split("\n") if not ln.strip().startswith("post_quantization_optimization(finetune")) \
            + "\n" + no_ft + "\n"

    calib = shape_for(P["calib_in"])
    rec["calib"] = {"n": int(len(calib)), "shape": list(calib.shape), "source": "coco_ret_exclude 캡션"}

    def optimize_with(alls, label):
        (LOG_DIR / f"{NET}{SFX}.{label}.alls").write_text(alls)
        runner.load_model_script(alls)
        runner.optimize(calib)
        runner.save_har(str(HAILO_DIR / f"{NET}{SFX}.quantized.har"))
        rec["alls_used"] = label

    if a.no_finetune:
        stage("optimize_no_finetune", lambda: optimize_with(alls_without_finetune(base_alls), "no_finetune"))
    else:
        ok = stage("optimize_official", lambda: optimize_with(base_alls, "official") or True, fatal=False)
        if not ok:
            print("[text-dfc] 공식 설정(finetune 포함) optimize 실패 -> finetune 만 끄고 다시 시도", flush=True)
            runner = ClientRunner(har=str(HAILO_DIR / f"{NET}{SFX}.parsed.har"), hw_arch=a.hw_arch)
            globals()["runner"] = runner
            def retry():
                (LOG_DIR / f"{NET}{SFX}.no_finetune.alls").write_text(alls_without_finetune(base_alls))
                runner.load_model_script(alls_without_finetune(base_alls))
                runner.optimize(calib)
                runner.save_har(str(HAILO_DIR / f"{NET}{SFX}.quantized.har"))
                rec["alls_used"] = "no_finetune (공식 설정 실패 후)"
            stage("optimize_no_finetune", retry)

    def do_eval():
        x = shape_for(P["check_in"]); eot = P["check_eot"]; ref = P["check_ref"]
        def run(ctx_kind):
            with runner.infer_context(ctx_kind) as ctx:
                out = np.asarray(runner.infer(ctx, x, batch_size=8))
            return post_numpy(out.reshape(len(x), 77, -1).astype(np.float32), eot, P["ln_w"], P["ln_b"], P["proj"],
                              ln_in_graph=bool(P["ln_in_graph"]) if "ln_in_graph" in P else True)
        fp = run(InferenceContext.SDK_FP_OPTIMIZED)
        q = run(InferenceContext.SDK_QUANTIZED)
        res = {"n_captions": int(len(x)),
               "fp_vs_open_clip_cos": {"mean": float(np.sum(fp * ref, 1).mean()), "min": float(np.sum(fp * ref, 1).min())},
               "quant_vs_open_clip_cos": {"mean": float(np.sum(q * ref, 1).mean()), "min": float(np.sum(q * ref, 1).min())}}
        img_of = P["check_img_of"]
        for code in ("A0", "H-fc2b7", "H-official"):
            f = C.RESULTS / "coco_ret" / f"{code}-select.npz"
            if not f.exists():
                continue
            im = np.load(f)["emb"]
            r1 = {k: float((np.argmax(t @ im.T, 1) == img_of).mean()) for k, t in
                  (("text_fp32_open_clip", ref), ("text_hailo_fp", fp), ("text_hailo_quant", q))}
            res[f"caption_R@1_gallery_{code}"] = r1
        np.savez_compressed(LOG_DIR / f"text_check{SFX}.npz", fp=fp, q=q, ref=ref, img_of=img_of)
        rec["check_eval"] = res
        print("[text-dfc] 간이 평가:", json.dumps(res, ensure_ascii=False), flush=True)
    stage("check_eval", do_eval, fatal=False)

    if not a.skip_compile:
        def do_compile():
            hef = runner.compile()
            hp = HAILO_DIR / f"{NET}{SFX}.{a.hw_arch}.hef"
            hp.write_bytes(hef)
            runner.save_har(str(HAILO_DIR / f"{NET}{SFX}.compiled.har"))
            rec["hef"] = {"path": str(hp), "size_mb": C.size_mb(hp), "sha256": C.sha256_file(hp)}
        stage("compile", do_compile)
    save()
    print("\n== 텍스트 인코더 요약")
    for k, v in rec["stages"].items():
        print(f"  {k:22s} {'성공' if v['ok'] else '실패'}  {v['seconds']}s")
    if "hef" in rec:
        print(f"  HEF: {rec['hef']['path']} ({rec['hef']['size_mb']} MB)")
    print(f"  로그: {log_path}")


if __name__ == "__main__":
    main()
