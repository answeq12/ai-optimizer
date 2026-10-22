"""사진 색인·검색 실측 (보고서 3.2 '왜 경량화하나' 근거). 사후 측정, 구성 선택과 무관

  python -m src.index_bench run            -> results/index_bench.md, results/index_bench.json

1) 색인: 사진 N장(기본 1,000)을 한 장씩 [JPEG 디코드 + 전처리 + 이미지 인코더] 순서로 처리.
   구성(A0 FP32 / A1 동적 / SPL-b567-k32 최종)마다 새 프로세스, 4스레드, 배치 1, 워밍업 20장 제외.
   잰 것: 전체 시간(1,000장), 사진/초, 인코더만의 지연 p50/p95, 프로세스 최대 상주 메모리(VmHWM).
   구성 순서를 바꿔 3회 반복, 중앙값 보고.
2) 검색: FP32 텍스트 인코더로 질의 1개를 임베딩 + 색인과 내적 + 상위 10장. COCO 캡션 300개, p50/p95.
   색인은 최종 구성(INT8)으로 만든 임베딩 = 'INT8 로 색인, FP32 텍스트로 검색' 실제 조합.
   내적+상위10 만의 시간은 색인 크기 1천/1만/10만 장(무작위 단위벡터)으로 따로 잰다.
- 사진은 COCO val2017 앞쪽 N장(파일명 순). 시간 측정 전용이며 정확도·선택에 쓰지 않는다
- 절대 시간은 이 PC(i9-14900KF, WSL) 기준. 기기마다 다르므로 FP32 대비 배율을 함께 본다
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
from . import common as C

CONFIGS = [("A0", "clip_vitb16_image_fp32.onnx"), ("A1", "a1_dynamic.onnx"), ("SPL-b567-k32", "imp/SPL-b567-k32.onnx")]
OUT = C.RESULTS / "index_bench"


def vmhwm() -> float:
    with open("/proc/self/status") as f:
        for ln in f:
            if ln.startswith("VmHWM:"):
                return int(ln.split()[1]) / 1024
    return float("nan")


def files(n: int) -> list[Path]:
    fs = sorted(C.COCO_DIR.glob("*.jpg"))
    if len(fs) < n + 20:
        sys.exit(f"사진이 부족합니다: {len(fs)}장")
    return fs[: n + 20]


def child_index(a) -> None:
    from PIL import Image
    from .backends import make_ort_session
    sess = make_ort_session(Path(a.model), a.threads); name = sess.get_inputs()[0].name
    fs = files(a.n)
    for f in fs[:20]:   # 워밍업
        sess.run(None, {name: C.preprocess(Image.open(f).convert("RGB"))[None]})
    enc = np.empty(a.n); emb = np.empty((a.n, 512), np.float32)
    t0 = time.perf_counter()
    for i, f in enumerate(fs[20:]):
        x = C.preprocess(Image.open(f).convert("RGB"))[None]
        t = time.perf_counter()
        e = sess.run(None, {name: x})[0][0]
        enc[i] = (time.perf_counter() - t) * 1000
        emb[i] = e / np.linalg.norm(e)
    total = time.perf_counter() - t0
    if a.save_emb:
        np.save(a.save_emb, emb)
    print(json.dumps({"total_s": total, "photos_per_s": a.n / total, "enc_p50_ms": float(np.percentile(enc, 50)),
                      "enc_p95_ms": float(np.percentile(enc, 95)), "enc_mean_ms": float(enc.mean()),
                      "pipeline_ms_per_photo": total * 1000 / a.n, "vmhwm_mb": vmhwm()}))


def child_search(a) -> None:
    import torch, open_clip
    from . import coco_retrieval as R
    torch.set_num_threads(a.threads)
    model = open_clip.create_model(C.MODEL_NAME, pretrained=C.PRETRAINED, force_quick_gelu=True, cache_dir=str(C.CACHE)).eval()
    tok = open_clip.get_tokenizer(C.MODEL_NAME)
    index = np.load(a.index)
    _, _, caps = R.captions_for("select")
    qs = caps[:: max(1, len(caps) // (a.q + 20))][: a.q + 20]
    lat, txt = [], []
    with torch.no_grad():
        for i, q in enumerate(qs):
            t = time.perf_counter()
            e = model.encode_text(tok([q])).float().numpy()[0]; e /= np.linalg.norm(e)
            s = index @ e; top = np.argpartition(-s, 10)[:10]; top = top[np.argsort(-s[top])]
            dt = (time.perf_counter() - t) * 1000
            if i >= 20:
                lat.append(dt)
    scan = {}
    rng = np.random.default_rng(0)
    for n in (1_000, 10_000, 100_000):
        g = rng.standard_normal((n, 512)).astype(np.float32); g /= np.linalg.norm(g, axis=1, keepdims=True)
        e = g[0]; ts = []
        for _ in range(200):
            t = time.perf_counter(); s = g @ e; top = np.argpartition(-s, 10)[:10]; ts.append((time.perf_counter() - t) * 1000)
        scan[n] = {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "index_mb_fp32": n * 512 * 4 / 1e6}
    print(json.dumps({"query_p50_ms": float(np.percentile(lat, 50)), "query_p95_ms": float(np.percentile(lat, 95)),
                      "n_queries": len(lat),
                      "scan": scan, "vmhwm_mb": vmhwm()}))


def spawn(args: list[str]) -> dict:
    r = subprocess.run([sys.executable, "-m", "src.index_bench", *args], capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().split("\n") if l.startswith("{")]
    if r.returncode or not lines:
        print(r.stdout[-2000:], r.stderr[-3000:]); raise SystemExit(f"✘ 자식 프로세스 실패: {args}")
    return json.loads(lines[-1])


def run(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    M = C.MODELS; res = {c: [] for c, _ in CONFIGS}
    orders = [[0, 1, 2], [2, 0, 1], [1, 2, 0]][: a.reps]
    for r, order in enumerate(orders):
        for k in order:
            code, rel = CONFIGS[k]
            extra = ["--save-emb", str(OUT / f"{code}-index.npy")] if r == 0 else []
            d = spawn(["child-index", "--model", str(M / rel), "--n", str(a.n), "--threads", str(a.threads), *extra])
            res[code].append(d)
            print(f"[index] 반복 {r+1}/{len(orders)} {code}: {a.n}장 {d['total_s']:.1f}s, {d['photos_per_s']:.1f}장/s, "
                  f"인코더 p50 {d['enc_p50_ms']:.2f} ms, 메모리 {d['vmhwm_mb']:.0f} MB", flush=True)
    med = {c: {k: float(np.median([d[k] for d in v])) for k in v[0]} for c, v in res.items()}
    s = spawn(["child-search", "--index", str(OUT / "SPL-b567-k32-index.npy"), "--q", str(a.q), "--threads", str(a.threads)])
    base = med["A0"]
    L = ["# 사진 색인·검색 실측 (사후 측정)", "", f"생성 {C.now()}. CPU: {cpu()}. 4스레드, 배치 1, 구성마다 새 프로세스, 순서를 바꿔 {len(orders)}회 반복의 중앙값", "",
         f"## 1. 색인: 사진 {a.n}장을 한 장씩 (JPEG 디코드 + 전처리 + 이미지 인코더)", "",
         f"| 구성 | {a.n}장 시간 (s) | 사진/초 | 1장당 전체 (ms) | 인코더만 p50 / p95 (ms) | 최대 메모리 (MB) | FP32 대비 처리량 |", "|---|---|---|---|---|---|---|"]
    names = {"A0": "FP32 (A0)", "A1": "동적 INT8 (A1, 비교 기준)", "SPL-b567-k32": "최종 INT8 (SPL-b567-k32)"}
    for c, _ in CONFIGS:
        m = med[c]
        L.append(f"| {names[c]} | {m['total_s']:.1f} | {m['photos_per_s']:.1f} | {m['pipeline_ms_per_photo']:.1f} | "
                 f"{m['enc_p50_ms']:.2f} / {m['enc_p95_ms']:.2f} | {m['vmhwm_mb']:.0f} | {m['photos_per_s']/base['photos_per_s']:.2f}배 |")
    f, q = med["A0"], med["SPL-b567-k32"]
    L += ["", f"- 사진 1만 장으로 환산(선형): FP32 {1e4/f['photos_per_s']/60:.1f}분 → 최종 INT8 {1e4/q['photos_per_s']/60:.1f}분",
          f"- 30 fps(1장 33.3 ms) 기준: 인코더만 p50 FP32 {f['enc_p50_ms']:.1f} ms → INT8 {q['enc_p50_ms']:.1f} ms, "
          f"디코드 포함 전체 FP32 {f['pipeline_ms_per_photo']:.1f} ms → INT8 {q['pipeline_ms_per_photo']:.1f} ms "
          f"({'INT8 은 30 fps 이내' if q['pipeline_ms_per_photo'] <= 33.3 else 'INT8 도 30 fps 미달'}, "
          f"{'FP32 는 30 fps 이내' if f['pipeline_ms_per_photo'] <= 33.3 else 'FP32 는 30 fps 미달'})",
          "", "## 2. 검색: 영어 질의 1개 (FP32 텍스트 인코더 + 내적 + 상위 10장)", "",
          f"- 색인 = 최종 INT8 로 만든 {a.n}장 임베딩, 질의 = COCO-select 캡션 {s['n_queries']}개 (앞 20개 워밍업 제외)",
          f"- 질의 1개 전체: p50 {s['query_p50_ms']:.1f} ms, p95 {s['query_p95_ms']:.1f} ms, 프로세스 최대 메모리 {s['vmhwm_mb']:.0f} MB",
          "", "| 색인 크기 | 내적+상위10 p50 / p95 (ms) | 색인 메모리 (FP32 512차원) |", "|---|---|---|"]
    for n, v in s["scan"].items():
        L.append(f"| {int(n):,}장 | {v['p50_ms']:.2f} / {v['p95_ms']:.2f} | {v['index_mb_fp32']:.1f} MB |")
    L += ["", "## 읽는 법", "", "- 비싼 건 색인(사진마다 이미지 인코더 1회)이고, 검색은 텍스트 인코더 1회 + 내적이라 색인 크기에 거의 무관하다",
          "- 사진은 COCO val2017 파일명 순 앞쪽 장. 시간 측정 전용(정확도·선택과 무관)",
          "- 절대 시간은 이 PC 기준. 다른 기기에서는 FP32 대비 배율로 본다. Hailo 수치가 아니다"]
    (C.RESULTS / "index_bench.md").write_text("\n".join(L) + "\n")
    (C.RESULTS / "index_bench.json").write_text(json.dumps({"index_median": med, "index_runs": res, "search": s, "n": a.n,
                                                            "threads": a.threads, "cpu": cpu(), "timestamp": C.now()}, indent=1, ensure_ascii=False))
    print("\n".join(L))


def cpu() -> str:
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        return "?"


def main() -> None:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--n", type=int, default=1000); p.add_argument("--reps", type=int, default=3)
    p.add_argument("--q", type=int, default=300); p.add_argument("--threads", type=int, default=C.DEFAULT_THREADS)
    p = sub.add_parser("child-index"); p.add_argument("--model"); p.add_argument("--n", type=int); p.add_argument("--threads", type=int); p.add_argument("--save-emb", default="")
    p = sub.add_parser("child-search"); p.add_argument("--index"); p.add_argument("--q", type=int); p.add_argument("--threads", type=int)
    a = ap.parse_args()
    {"run": run, "child-index": child_index, "child-search": child_search}[a.cmd](a)


if __name__ == "__main__":
    main()
