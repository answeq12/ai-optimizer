"""추론만 하는 측정용 자식 프로세스. benchmark.py 가 구성마다 새로 띄운다.

변환·양자화·평가와 프로세스를 분리해야 그때 잡은 메모리가 섞이지 않는다.
torch 와 데이터셋을 불러오지 않고 numpy + onnxruntime 만 쓴다.

출력(JSON 한 줄): 지연시간 분위수(ms)와 프로세스 최대 상주 메모리.
max_rss_mb 는 /proc/self/status 의 VmHWM(이 프로세스의 상주 메모리 최고치)이다.
getrusage 의 ru_maxrss 는 리눅스에서 부모 프로세스의 최고치를 물려받는다. 부모(benchmark.py)가
자식보다 메모리를 많이 쓰면 모든 구성이 같은 값으로 찍힌다(실제로 겪음). 그래서 exec 때 초기화되는
VmHWM 을 쓴다. VmHWM 을 읽을 수 없을 때만 ru_maxrss 로 대신한다.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time

import numpy as np


def peak_rss_mb() -> tuple[float, str]:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0, "VmHWM"
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, "ru_maxrss"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--warmup", type=int, required=True)
    ap.add_argument("--runs", type=int, required=True)
    args = ap.parse_args()

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    sess = ort.InferenceSession(args.model, sess_options=so, providers=["CPUExecutionProvider"])
    load_s = time.perf_counter() - t0
    name = sess.get_inputs()[0].name
    x = np.load(args.input).astype(np.float32)

    for _ in range(args.warmup):
        sess.run(None, {name: x})
    lat = np.empty(args.runs, dtype=np.float64)
    for i in range(args.runs):
        t = time.perf_counter_ns()
        sess.run(None, {name: x})
        lat[i] = (time.perf_counter_ns() - t) / 1e6
    lat.sort()
    rss_mb, rss_src = peak_rss_mb()
    json.dump({
        "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
        "mean_ms": float(lat.mean()), "min_ms": float(lat[0]), "max_ms": float(lat[-1]),
        "max_rss_mb": rss_mb, "rss_source": rss_src, "load_s": load_s,
        "runs": args.runs, "warmup": args.warmup, "threads": args.threads,
    }, sys.stdout)


if __name__ == "__main__":
    main()
