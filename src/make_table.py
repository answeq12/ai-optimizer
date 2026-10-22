"""결과 표 생성.

  python -m src.make_table            # 첫 비교표 (A0·A1·A2) -> results/first_table.md
  python -m src.make_table --all      # 정확도를 잰 모든 구성 -> results/all_runs.md

정확도는 results/preds/<run_id>.json, 속도·메모리는 results/latency_summary.json,
정수 연산 비율은 results/coverage.json 에서 읽는다. 값이 없는 칸은 '-' 로 둔다.
"""
from __future__ import annotations

import argparse
import csv
import json

from . import common as C

NAMES = {"A0": "FP32", "A1": "동적 INT8", "A2": "정적 INT8 기본"}


def _load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def fmt(v, spec, suffix=""):
    return "-" if v is None else format(v, spec) + suffix


def build(codes: list[str], eval_set: str, title: str) -> str:
    lat = _load(C.RESULTS / "latency_summary.json", {}).get("configs", {})
    cov = _load(C.RESULTS / "coverage.json", {})
    acc = {c: _load(C.RESULTS / "preds" / f"{c}-{eval_set}.json", None) for c in codes}
    base = acc.get("A0")
    L = [f"# {title}", "",
         "| 구성 | top-1 | Δ vs FP32 | FP32 대비 일치율 | p50 (ms) | p95 (ms) | Max RSS (MB) | 파일 (MB) | 정수 연산 비율 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in codes:
        a, l, v = acc.get(c), lat.get(c, {}), cov.get(c, {})
        top1 = a["top1_acc"] * 100 if a else None
        delta = (a["top1_acc"] - base["top1_acc"]) * 100 if a and base and c != "A0" else None
        agree = a.get("agree_with_fp32") * 100 if a and a.get("agree_with_fp32") is not None else None
        L.append("| " + " | ".join([
            f"{c} {NAMES.get(c, '')}".strip(),
            fmt(top1, ".2f", "%"),
            "—" if c == "A0" else fmt(delta, "+.2f", "%p"),
            "—" if c == "A0" else fmt(agree, ".2f", "%"),
            fmt(l.get("p50_ms"), ".2f"), fmt(l.get("p95_ms"), ".2f"),
            fmt(l.get("max_rss_mb"), ".0f"), fmt(l.get("model_size_mb"), ".1f"),
            fmt(v.get("quantized_op_ratio") * 100 if v else None, ".1f", "%"),
        ]) + " |")
    L.append("")
    if base:
        L.append(f"- 평가셋: CIFAR-100 {eval_set} {base['n']}장. top-1 표준오차 약 ±{base['std_err']*100:.2f}%p. "
                 "이 크기 이내의 차이만으로 방향을 확정하지 않는다")
    for c in codes:
        a = acc.get(c)
        if a and a.get("mcnemar_vs_fp32"):
            m = a["mcnemar_vs_fp32"]
            L.append(f"- {c} vs FP32 짝지은 비교: FP32만 정답 {m['a_only']}장, {c}만 정답 {m['b_only']}장, "
                     f"McNemar p={m['p_value']:.3g}")
    any_lat = next((lat[c] for c in codes if c in lat), None)
    if any_lat:
        s = _load(C.RESULTS / "latency_summary.json", {})
        cpu = s.get("cpu", {})
        L.append(f"- 속도·메모리: 배치 1, 스레드 {any_lat['threads']}, 워밍업 {any_lat['warmup']}회 제외 후 "
                 f"{any_lat['runs']}회, {any_lat['repeats']}회 반복의 중앙값. "
                 f"측정 환경 {cpu.get('model_name', '?')} / WSL={s.get('is_wsl')} / env {s.get('env_hash')}")
        if "A0" in lat:
            for c in codes:
                if c != "A0" and c in lat:
                    L.append(f"- {c}: p50 기준 FP32 대비 {lat['A0']['p50_ms']/lat[c]['p50_ms']:.2f}배 빠름, "
                             f"파일 크기 {lat['A0']['model_size_mb']/lat[c]['model_size_mb']:.2f}배 작음")
    if (C.DATA / "FAKE_DATA").exists():
        L.append("- **경고: 가짜 데이터로 만든 표입니다. 수치는 무의미합니다.**")
    L += ["", "이 표로 말할 수 없는 것: Hailo 에서의 개선량(양자화기가 다르다), 제조사 설정 대비 우위, "
          "다른 도메인에서의 성능, 최종 점수 전망.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--eval-set", default="val")
    args = ap.parse_args()
    if args.all:
        codes, seen = [], set()
        p = C.RESULTS / "accuracy.csv"
        if p.exists():
            for r in csv.DictReader(open(p, encoding="utf-8")):
                rid = r["run_id"]
                if r["eval_set"] == args.eval_set and rid.endswith("-" + args.eval_set):
                    c = rid[: -len(args.eval_set) - 1]
                    if c not in seen:
                        seen.add(c)
                        codes.append(c)
        out, title = C.RESULTS / "all_runs.md", f"전체 구성 ({args.eval_set})"
    else:
        codes, out, title = ["A0", "A1", "A2"], C.RESULTS / "first_table.md", "첫 비교표 (파일럿)"
    text = build(codes, args.eval_set, title)
    out.write_text(text, encoding="utf-8")
    print(text)
    print("[table] 저장:", out)


if __name__ == "__main__":
    main()
