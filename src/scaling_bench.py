"""스레드 수·배치 크기별 속도 (사후 측정, 보고서 6.5 보강). PC 가 조용할 때만

  python -m src.scaling_bench        -> results/scaling_bench.md, .json

- 구성: FP32(A0), 최종 INT8(SPL-b567-k32). 스레드 1/2/4 x 배치 1, 그리고 스레드 4 x 배치 8(입력 배치가 동적일 때만)
- 측정은 bench_child(구성마다 새 프로세스, VmHWM)로, 워밍업 10회 + 100회, 순서를 바꿔 3회 반복 중앙값
- 질문: 코어가 적은 엣지 CPU 조건(1~2스레드)과 사진을 묶어 처리하는 색인 조건(배치 8)에서도 INT8 이득이 유지되나
"""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from . import common as C

MODELS = [("A0", "clip_vitb16_image_fp32.onnx"), ("SPL-b567-k32", "imp/SPL-b567-k32.onnx")]


def dynamic_batch(path: Path) -> bool:
    import onnxruntime as ort
    d = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).get_inputs()[0].shape[0]
    return not isinstance(d, int)


def run_child(model: Path, x: Path, threads: int, warmup: int, runs: int) -> dict:
    r = subprocess.run([sys.executable, "-m", "src.bench_child", "--model", str(model), "--input", str(x),
                        "--threads", str(threads), "--warmup", str(warmup), "--runs", str(runs)], capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().split("\n") if l.startswith("{")]
    if r.returncode or not lines:
        print(r.stdout[-1500:], r.stderr[-2500:]); raise SystemExit("✘ bench_child 실패")
    return json.loads(lines[-1])


def main() -> None:
    M = C.MODELS
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp())
    xs = {}
    for b in (1, 8):
        p = tmp / f"x{b}.npy"; np.save(p, rng.standard_normal((b, 3, 224, 224)).astype(np.float32)); xs[b] = p
    dyn = all(dynamic_batch(M / rel) for _, rel in MODELS)
    conds = [(1, 1), (2, 1), (4, 1)] + ([(4, 8)] if dyn else [])
    res = {}
    for rep in range(3):
        order = MODELS if rep % 2 == 0 else MODELS[::-1]
        for th, b in conds:
            for code, rel in order:
                d = run_child(M / rel, xs[b], th, 10, 100)
                res.setdefault((code, th, b), []).append(d)
                print(f"[scale] 반복 {rep+1}/3 {code} 스레드 {th} 배치 {b}: p50 {d['p50_ms']:.2f} ms, 메모리 {d.get('max_rss_mb', 0):.0f} MB", flush=True)
    med = {k: {"p50_ms": float(np.median([d["p50_ms"] for d in v])), "p95_ms": float(np.median([d["p95_ms"] for d in v])),
               "rss_mb": float(np.median([d.get("max_rss_mb", 0) for d in v]))} for k, v in res.items()}
    L = ["# 스레드·배치별 속도 (사후 측정)", "", f"생성 {C.now()}. CPU {cpu()}. 워밍업 10 + 100회, 3반복 중앙값, 구성마다 새 프로세스", "",
         "| 스레드 | 배치 | FP32 사진 1장당 (ms) | 최종 INT8 사진 1장당 (ms) | FP32 대비 | INT8 메모리 (MB) |", "|---|---|---|---|---|---|"]
    for th, b in conds:
        f = med[("A0", th, b)]["p50_ms"] / b; q = med[("SPL-b567-k32", th, b)]["p50_ms"] / b
        L.append(f"| {th} | {b} | {f:.2f} | {q:.2f} | {f/q:.2f}배 | {med[('SPL-b567-k32', th, b)]['rss_mb']:.0f} |")
    if not dyn:
        L += ["", "- 모델 입력의 배치 차원이 1로 고정돼 있어 배치 8 은 재지 않았다"]
    L += ["", "- 사진 1장당 = p50 / 배치. 배치 1 은 지연, 배치 8 은 처리량 관점", "- 절대값은 이 PC 기준. 배율로 본다. Hailo 수치 아님"]
    (C.RESULTS / "scaling_bench.md").write_text("\n".join(L) + "\n")
    (C.RESULTS / "scaling_bench.json").write_text(json.dumps({f"{k[0]}|t{k[1]}|b{k[2]}": v for k, v in med.items()}, indent=1))
    print("\n".join(L))


def cpu() -> str:
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except Exception:
        return "?"


if __name__ == "__main__":
    main()
