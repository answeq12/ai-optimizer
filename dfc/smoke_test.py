"""Hailo DFC 스모크 테스트 (P5). 정식 실험 전에 "이 경로가 굴러가는가"만 최소 비용으로 판정한다.

  python -m dfc.smoke_test                 # parse -> 64장 optimize -> 에뮬레이터 평가 300장
  python -m dfc.smoke_test --compile       # 위에 더해 Hailo-10H 용 HEF 컴파일까지
  python -m dfc.smoke_test --onnx ours     # Hailo 공식 ONNX 대신 우리가 export 한 ONNX 로

DFC 전용 파이썬 환경(~/venvs/hailo)에서 실행한다. 파일럿 환경과 합치지 않는다(numpy·onnx 버전 충돌).

단계마다 소요 시간과 성공 여부를 logs/dfc/smoke_<시각>.json 에 남긴다. 실패해도 로그를 지우지 않는다.

입력 규약 주의: Hailo CLIP 은 normalize_in_net=true. 캘리브레이션과 추론 입력은 정규화 전 0~255 값이다.
모델 스크립트(.alls)는 Hailo Model Zoo 가 이 모델에 쓰는 공식 설정을 그대로 쓴다
(optimization_level=0, calibset_size=64, 일부 층 a16_w16).

이 스크립트는 DFC 5.4.0 의 소스(ClientRunner)를 읽고 작성했으며, 작성 시점에 DFC 를 실제로 실행해 보지는 못했다.
"""
from __future__ import annotations

import argparse
import os
import glob
import json
import re
import sys
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from src import common as C

NET = "clip_vit_b_16_image_encoder"
HAILO_DIR = C.STORE / "hailo"
LOG_DIR = C.LOGS / "dfc"


def zoo_wheel() -> Path:
    # installers/ 우선, 예전 위치(프로젝트 루트)도 찾는다
    c = sorted(glob.glob(str(C.ROOT / "installers" / "hailo_model_zoo-*.whl"))) \
        or sorted(glob.glob(str(C.ROOT / "hailo_model_zoo-*.whl")))
    if not c:
        sys.exit("installers/ 폴더에 hailo_model_zoo-*.whl 이 없습니다.")
    return Path(c[-1])


def zoo_read(name_suffix: str) -> str:
    """Model Zoo 휠을 설치하지 않고 설정 파일만 꺼내 읽는다."""
    with zipfile.ZipFile(zoo_wheel()) as z:
        hit = [n for n in z.namelist() if n.endswith(name_suffix)]
        if not hit:
            raise FileNotFoundError(name_suffix)
        return z.read(hit[0]).decode()


def official_onnx() -> Path:
    """Model Zoo 설정에 적힌 공식 ONNX 를 내려받는다."""
    cfg = zoo_read(f"cfg/networks/{NET}.yaml")
    url = next(ln.split("url:", 1)[1].strip() for ln in cfg.split("\n") if ln.strip().startswith("url:"))
    d = HAILO_DIR / "official"
    d.mkdir(parents=True, exist_ok=True)
    hit = list(d.glob("*.onnx"))
    if hit:
        return hit[0]
    zp = d / Path(url).name
    print(f"[dfc] 공식 ONNX 다운로드: {url}")
    urllib.request.urlretrieve(url, zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(d)
    hit = list(d.rglob("*.onnx"))
    if not hit:
        raise FileNotFoundError("zip 안에 onnx 가 없습니다")
    return hit[0]


def val_uint8(n: int):
    data, labels = C.load_eval_set("val")
    step = max(1, len(data) // n)          # 클래스가 고르게 섞이도록 일정 간격으로 뽑는다
    idx = np.arange(0, len(data), step)[:n]
    x = np.stack([C.preprocess_uint8(Image.fromarray(data[i])) for i in idx]).astype(np.float32)
    return x, labels[idx]


def embed(runner, ctx_kind, x: np.ndarray) -> np.ndarray:
    with runner.infer_context(ctx_kind) as ctx:
        out = runner.infer(ctx, x, batch_size=8)
    e = np.asarray(out).reshape(len(x), -1).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def hn_layers(runner) -> dict:
    """파싱된 모델의 HN 층 목록 {전체이름: 층 정보}. DFC 버전마다 접근자가 달라 몇 가지를 시도한다."""
    for getter in ("get_hn_dict", "get_hn"):
        if hasattr(runner, getter):
            hn = getattr(runner, getter)()
            if isinstance(hn, str):
                hn = json.loads(hn)
            elif not isinstance(hn, dict) and hasattr(hn, "to_hn"):
                hn = json.loads(hn.to_hn(hn.name)) if callable(hn.to_hn) else hn
            if isinstance(hn, dict) and "layers" in hn:
                return hn["layers"]
    raise RuntimeError("HN 층 목록을 읽지 못했습니다 (get_hn_dict / get_hn)")


def _last_dim(shapes):
    try:
        s = shapes[0] if shapes and isinstance(shapes[0], (list, tuple)) else shapes
        return int(s[-1])
    except Exception:
        return None


def find_fc2_layers(runner, blocks: list[int], dump: Path) -> tuple[list[str], str]:
    """블록별 MLP 축소층(fc2, 3072 -> 768)의 Hailo 층 이름을 찾는다.

    1순위: 원래 ONNX 이름(original_names)의 resblocks.N/mlp/c_proj
    2순위: 입력 마지막 차원 3072, 출력 768 인 층을 그래프 순서대로 (정확히 12개일 때만)
    둘 다 12개가 안 되면 멈춘다. 추측으로 이어가지 않는다. 층 목록은 dump 에 저장.
    """
    rows = []
    for full, L in hn_layers(runner).items():
        rows.append({"name": full.split("/")[-1], "full": full, "type": L.get("type"),
                     "original_names": L.get("original_names", []),
                     "in": L.get("input_shapes"), "out": L.get("output_shapes")})
    dump.write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str))
    rx = re.compile(r"resblocks[._/](\d+)[./_]mlp[./_]c_proj")
    by_block: dict[int, list[str]] = {}
    for r in rows:
        for o in r["original_names"] or []:
            m = rx.search(str(o))
            if m:
                by_block.setdefault(int(m.group(1)), [])
                if r["name"] not in by_block[int(m.group(1))]:
                    by_block[int(m.group(1))].append(r["name"])
    if len(by_block) == 12:
        return [n for b in blocks for n in by_block[b]], "original_names"
    cand = [r["name"] for r in rows if _last_dim(r["in"]) == 3072 and _last_dim(r["out"]) == 768]
    if len(cand) == 12:
        return [cand[b] for b in blocks], "shape 3072->768 (graph order)"
    raise RuntimeError(f"fc2 층을 특정하지 못했습니다: 이름 매칭 {len(by_block)}블록, 형상 매칭 {len(cand)}개 "
                       f"(기대 12). 층 목록: {dump}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="official", help="official | ours | ONNX 파일 경로")
    ap.add_argument("--hw-arch", default="hailo10h")
    ap.add_argument("--calib-list", default=str(C.SPLITS / "calib_coco_64_s0.txt"))
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--compile", action="store_true", help="HEF 컴파일까지 (오래 걸릴 수 있다)")
    ap.add_argument("--tag", default="", help="산출물(HAR·HEF) 이름표. 다른 설정으로 만든 HEF 가 서로 덮어쓰지 않게 한다")
    ap.add_argument("--fc2-16bit-blocks", default="",
                    help="쉼표 구분 블록 번호. 그 블록의 fc2(3072->768)를 a16_w16 으로 남긴다 (ORT 민감도 결과의 이식)")
    ap.add_argument("--fc2-precision", default="a16_w16", choices=["a16_w16", "a16_w8"],
                    help="--fc2-16bit-blocks 로 지정한 fc2 의 정밀도. a16_w8 = 활성값만 16비트, 가중치는 8비트")
    ap.add_argument("--from-har", default="",
                    help="이미 양자화된 HAR 을 불러와 평가만 한다 (parse/optimize 생략)")
    ap.add_argument("--save-preds", action="store_true", help="이미지별 예측을 logs/dfc/preds_*.npz 로 저장 (짝지은 비교용)")
    args = ap.parse_args()
    SFX = f".{args.tag}" if args.tag else ""

    C.ensure_dirs()
    HAILO_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rec: dict = {"timestamp": C.now(), "net": NET, "hw_arch": args.hw_arch, "args": vars(args),
                 "stages": {}, "env": C.env_info()}
    log_path = LOG_DIR / f"smoke_{stamp}{SFX}.json"

    def save():
        log_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))

    def stage(name, fn):
        print(f"[dfc] ▶ {name}", flush=True)
        t0 = time.time()
        try:
            out = fn()
            rec["stages"][name] = {"ok": True, "seconds": round(time.time() - t0, 1)}
            print(f"[dfc] ✔ {name} ({rec['stages'][name]['seconds']}s)", flush=True)
            save()
            return out
        except Exception as e:                       # 실패도 기록하고 멈춘다
            rec["stages"][name] = {"ok": False, "seconds": round(time.time() - t0, 1),
                                   "error": f"{type(e).__name__}: {e}",
                                   "traceback": traceback.format_exc()[-6000:]}
            rec["failed_stage"] = name
            save()
            print(f"[dfc] ✘ {name} 실패: {type(e).__name__}: {str(e)[:500]}")
            print(f"[dfc] 로그: {log_path}")
            sys.exit(1)

    def do_import():
        import hailo_sdk_client
        from hailo_sdk_client import ClientRunner, InferenceContext
        rec["dfc_version"] = getattr(hailo_sdk_client, "__version__", "?")
        try:
            import tensorflow as tf
            rec["tensorflow"] = tf.__version__
            rec["tf_gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
        except Exception as e:
            rec["tf_gpus"] = f"확인 실패: {e}"
        return ClientRunner, InferenceContext

    ClientRunner, InferenceContext = stage("import", do_import)
    print(f"[dfc] DFC {rec.get('dfc_version')}, TensorFlow GPU: {rec.get('tf_gpus')}")

    def do_onnx():
        if args.onnx == "official":
            p = official_onnx()
        elif args.onnx == "ours":
            p = C.RAW_ONNX
        else:
            p = Path(args.onnx)
        if not p.exists():
            raise FileNotFoundError(p)
        rec["onnx"] = {"path": str(p), "size_mb": C.size_mb(p), "sha256": C.sha256_file(p)}
        return p

    if args.from_har:
        def do_load():
            p = Path(args.from_har)
            if not p.exists():
                raise FileNotFoundError(p)
            rec["from_har"] = {"path": str(p), "size_mb": C.size_mb(p), "sha256": C.sha256_file(p)}
            return ClientRunner(har=str(p), hw_arch=args.hw_arch)
        runner = stage("load_har", do_load)
    else:
        runner = None
    onnx_path = stage("onnx", do_onnx) if runner is None else None

    def do_parse():
        r = ClientRunner(hw_arch=args.hw_arch)
        r.translate_onnx_model(str(onnx_path), NET)
        r.save_har(str(HAILO_DIR / f"{NET}{SFX}.parsed.har"))
        return r

    if runner is None:
        runner = stage("parse", do_parse)

    def do_script():
        alls = zoo_read(f"cfg/alls/generic/{NET}.alls")
        (LOG_DIR / f"{NET}.alls").write_text(alls)
        rec["alls_source"] = f"{zoo_wheel().name}: cfg/alls/generic/{NET}.alls"
        if args.fc2_16bit_blocks:
            blocks = [int(b) for b in args.fc2_16bit_blocks.split(",")]
            names, how = find_fc2_layers(runner, blocks, LOG_DIR / f"layers{SFX}.json")
            extra = f"quantization_param([{', '.join(names)}], precision_mode={args.fc2_precision})"
            alls = alls.rstrip("\n") + "\n" + extra + "\n"
            rec["fc2_16bit"] = {"blocks": blocks, "layers": names, "resolved_by": how, "alls_line": extra}
            print(f"[dfc] fc2 16비트: 블록 {blocks} -> {names} ({how})", flush=True)
        _cs = os.environ.get("HALIO_CALIBSET_SIZE")                  # 10/18: 공식 alls 의 calibset_size=64 를 바꿀 때만
        if _cs:
            assert "calibset_size=64" in alls, "alls 에 calibset_size=64 가 없습니다"
            alls = alls.replace("calibset_size=64", f"calibset_size={int(_cs)}")
        (LOG_DIR / f"{NET}{SFX}.alls").write_text(alls)
        runner.load_model_script(alls)

    if not args.from_har:
        stage("model_script", do_script)

    def do_optimize():
        calib = C.load_calib_images_uint8(Path(args.calib_list)).astype(np.float32)
        rec["calib"] = {"list": Path(args.calib_list).name, "n": int(len(calib)),
                        "sha256": C.sha256_file(args.calib_list), "input": "uint8 0~255 NHWC (정규화는 네트워크 안)"}
        runner.optimize(calib)
        runner.save_har(str(HAILO_DIR / f"{NET}{SFX}.quantized.har"))

    if not args.from_har:
        stage("optimize", do_optimize)

    def do_eval():
        x, labels = val_uint8(args.eval_n)
        fp = embed(runner, InferenceContext.SDK_FP_OPTIMIZED, x)
        q = embed(runner, InferenceContext.SDK_QUANTIZED, x)
        res = {"n": int(len(x)), "eval_set": "val 일부(간이)",
               "mean_cosine_fp_vs_quantized": float(np.mean(np.sum(fp * q, axis=1))),
               "min_cosine_fp_vs_quantized": float(np.min(np.sum(fp * q, axis=1)))}
        if C.TEXT_EMB.exists():
            t = np.load(C.TEXT_EMB)
            pf, pq = np.argmax(fp @ t.T, axis=1), np.argmax(q @ t.T, axis=1)
            res.update({"top1_fp_H0": float((pf == labels).mean()),
                        "top1_quantized_H1": float((pq == labels).mean()),
                        "agree_fp_vs_quantized": float((pf == pq).mean()),
                        "std_err_pp": float(100 * np.sqrt((pq == labels).mean() * (1 - (pq == labels).mean()) / len(x)))})
            if args.save_preds:
                pp = LOG_DIR / f"preds{SFX}_n{len(x)}.npz"
                np.savez_compressed(pp, fp=pf, q=pq, labels=labels, cos=np.sum(fp * q, axis=1))
                res["preds_file"] = str(pp)
        else:
            res["note"] = "텍스트 임베딩이 없어 정확도는 계산하지 않음 (먼저 파일럿 실행)"
        rec["emulator_eval"] = res
        return res

    res = stage("emulator_eval", do_eval)

    if args.compile:
        def do_compile():
            hef = runner.compile()
            hp = HAILO_DIR / f"{NET}{SFX}.{args.hw_arch}.hef"
            hp.write_bytes(hef)
            runner.save_har(str(HAILO_DIR / f"{NET}{SFX}.compiled.har"))
            rec["hef"] = {"path": str(hp), "size_mb": C.size_mb(hp), "sha256": C.sha256_file(hp)}
        stage("compile", do_compile)

    save()
    print("\n== DFC 스모크 테스트 요약")
    for k, v in rec["stages"].items():
        print(f"  {k:14s} {'성공' if v['ok'] else '실패'}  {v['seconds']}s")
    if "top1_fp_H0" in res:
        print(f"  에뮬레이터 정확도({res['n']}장, 간이): FP {res['top1_fp_H0']*100:.1f}% -> "
              f"양자화 {res['top1_quantized_H1']*100:.1f}%, 일치율 {res['agree_fp_vs_quantized']*100:.1f}%")
    print(f"  FP 대비 양자화 임베딩 코사인: 평균 {res['mean_cosine_fp_vs_quantized']:.4f}, "
          f"최소 {res['min_cosine_fp_vs_quantized']:.4f}")
    if "hef" in rec:
        print(f"  HEF: {rec['hef']['path']} ({rec['hef']['size_mb']} MB)")
    print(f"  로그: {log_path}")
    print("  주의: 300장 간이 평가의 표준오차는 약 ±2.7%p 다. 굴러가는지 확인하는 용도이며 결론 수치가 아니다.")


if __name__ == "__main__":
    main()
