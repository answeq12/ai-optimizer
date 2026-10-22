"""레이어 민감도 탐침. A2m(정적 INT8, 가중치 MatMul 49개) 의 손실이 어느 행렬곱에서 오는지 찾는다.

방법: 49개 중 한 묶음만 FP32 로 남기고(양자화에서 제외) 나머지는 A2m 과 똑같이 양자화한다.
그 묶음을 살렸을 때 정확도가 얼마나 돌아오는지(A2m 대비 회복량)가 그 묶음의 민감도다.
범위 추정·캘리브레이션·per-tensor 설정은 A2m 과 같다. 바뀌는 변수는 "어느 노드를 제외했나" 하나다.

  python -m src.sensitivity groups                 # 49개를 (블록, 종류)로 분류해 확인 -> results/sensitivity/groups.json
  python -m src.sensitivity names kind:fc2         # 해당 묶음의 노드 이름 (quantize_ort --exclude-nodes-file 용)
  python -m src.sensitivity names depth:0-3
  python -m src.sensitivity names block:5/fc2
  python -m src.sensitivity table                  # S-* 결과를 A2m 기준으로 정리 -> results/sensitivity.md

종류(kind)는 이름이 아니라 그래프 구조로 정한다. 블록 안에서 그래프 순서대로
  qkv   입력 768 -> 출력 2304 (어텐션 입력 투영, q·k·v 합본)
  out   768 -> 768 (어텐션 출력 투영)
  fc1   768 -> 3072 (MLP 확장)
  fc2   3072 -> 768 (MLP 축소, QuickGELU 직후 입력)
블록 밖의 마지막 하나는 proj (768 -> 512, 최종 임베딩 투영).
분류가 12블록 x 4종 + proj 1개로 떨어지지 않으면 멈춘다. 추측으로 이어가지 않는다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import common as C
from .evaluate import mcnemar
from .quantize_ort import preprocess_model, weight_matmul_nodes

OUT_DIR = C.RESULTS / "sensitivity"
GROUPS_JSON = OUT_DIR / "groups.json"
KINDS = ("qkv", "out", "fc1", "fc2")
BASE_CODE = "A2m"


def _weight_shape(m, node, inits) -> tuple[int, int]:
    """가중치 입력의 (입력 차원, 출력 차원). Gemm transB 도 처리."""
    for k, name in enumerate(node.input):
        if name in inits:
            dims = list(inits[name].dims)
            break
    else:
        raise RuntimeError(f"{node.name}: 가중치 initializer 를 찾지 못했습니다")
    if len(dims) != 2:
        raise RuntimeError(f"{node.name}: 가중치가 2차원이 아닙니다 {dims}")
    trans_b = any(a.name == "transB" and a.i == 1 for a in node.attribute) if node.op_type == "Gemm" else False
    if k == 0:                     # 상수가 왼쪽에 오는 경우는 예상하지 않는다
        raise RuntimeError(f"{node.name}: 가중치가 첫 번째 입력입니다. 분류 규칙을 확인하세요")
    return (dims[1], dims[0]) if trans_b else (dims[0], dims[1])


def classify() -> list[dict]:
    import onnx
    src = preprocess_model()
    m = onnx.load(str(src), load_external_data=False)
    inits = {i.name: i for i in m.graph.initializer}
    for n in m.graph.node:          # Constant 노드로 들어온 가중치도 같이 본다
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    inits[n.output[0]] = a.t
    wanted = set(weight_matmul_nodes(src))
    order = [n for n in m.graph.node if n.name in wanted]     # 그래프(위상) 순서
    rows = []
    for idx, n in enumerate(order):
        mb = re.search(r"resblocks[./_](\d+)", n.name)
        rows.append({"name": n.name, "op": n.op_type, "order": idx,
                     "block": int(mb.group(1)) if mb else None,
                     "shape": list(_weight_shape(m, n, inits))})

    # 이름에 블록 번호가 없으면 그래프 순서로 4개씩 묶는다 (마지막 1개는 proj)
    if any(r["block"] is None for r in rows[:-1]):
        print("[sens] 노드 이름에서 블록 번호를 못 읽어 그래프 순서로 4개씩 묶습니다")
        for r in rows[:-1]:
            r["block"] = r["order"] // 4
        rows[-1]["block"] = None

    by_block: dict[int, list[dict]] = {}
    for r in rows:
        if r["block"] is None:
            r["kind"] = "proj"
        else:
            by_block.setdefault(r["block"], []).append(r)
    for b, rs in by_block.items():
        rs.sort(key=lambda r: r["order"])
        sig = [tuple(r["shape"]) for r in rs]
        if len(rs) != 4:
            sys.exit(f"블록 {b}: 가중치 MatMul 이 {len(rs)}개입니다 (기대 4). 분류 중단. {[r['name'] for r in rs]}")
        expect = [(768, 2304), (768, 768), (768, 3072), (3072, 768)]
        if sig != expect:
            sys.exit(f"블록 {b}: 형상 {sig} 가 기대 {expect} 와 다릅니다. 분류 중단.")
        for r, k in zip(rs, KINDS):
            r["kind"] = k
    proj = [r for r in rows if r["kind"] == "proj"]
    if len(by_block) != 12 or len(proj) != 1 or len(rows) != 49:
        sys.exit(f"분류 결과가 12블록 x 4 + proj 1 이 아닙니다: 블록 {len(by_block)}, proj {len(proj)}, 전체 {len(rows)}")
    return rows


def select(rows: list[dict], spec: str) -> list[str]:
    """kind:fc2 / kind:qkv,out / depth:0-3 / block:5/fc2 / block:5 / blocks:6,7/fc2 / proj"""
    if spec == "proj":
        sel = [r for r in rows if r["kind"] == "proj"]
    elif spec.startswith("kind:"):
        ks = spec[5:].split(",")
        bad = [k for k in ks if k not in KINDS + ("proj",)]
        if bad:
            sys.exit(f"알 수 없는 종류: {bad}")
        sel = [r for r in rows if r["kind"] in ks]
    elif spec.startswith("depth:"):
        a, b = map(int, spec[6:].split("-"))
        sel = [r for r in rows if r["block"] is not None and a <= r["block"] <= b]
    elif spec.startswith("blocks:"):          # blocks:6,7/fc2 -> 여러 블록의 같은 종류
        body, _, kind = spec[7:].partition("/")
        bs = {int(x) for x in body.split(",")}
        sel = [r for r in rows if r["block"] in bs and (not kind or r["kind"] == kind)]
    elif spec.startswith("block:"):
        body = spec[6:]
        blk, _, kind = body.partition("/")
        sel = [r for r in rows if r["block"] == int(blk) and (not kind or r["kind"] == kind)]
    else:
        sys.exit(f"알 수 없는 선택: {spec}")
    if not sel:
        sys.exit(f"선택된 노드가 없습니다: {spec}")
    return [r["name"] for r in sel]


def load_rows() -> list[dict]:
    if GROUPS_JSON.exists():
        return json.loads(GROUPS_JSON.read_text())["nodes"]
    return classify()


def cmd_groups() -> None:
    rows = classify()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    GROUPS_JSON.write_text(json.dumps({"source": preprocess_model().name, "timestamp": C.now(),
                                       "nodes": rows}, indent=2, ensure_ascii=False))
    print(f"[sens] 가중치 MatMul {len(rows)}개 분류 완료 -> {GROUPS_JSON}")
    for k in KINDS + ("proj",):
        ex = [r for r in rows if r["kind"] == k]
        print(f"  {k:5s} {len(ex):2d}개  형상 {ex[0]['shape']}  예: {ex[0]['name']}")


def _acc(code: str):
    p = C.RESULTS / "preds" / f"{code}-val.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["preds"] == z["labels"], z["preds"]


def cmd_table() -> None:
    base = _acc(BASE_CODE)
    fp = _acc("A0")
    if base is None or fp is None:
        sys.exit("A0-val / A2m-val 예측이 없습니다. 먼저 파일럿과 a2fix 를 돌리세요.")
    gap = fp[0].mean() - base[0].mean()
    files = sorted((C.RESULTS / "preds").glob("S-*-val.npz"), key=lambda p: p.stat().st_mtime)
    codes = [p.name[:-len("-val.npz")] for p in files]                 # 실행 순서대로
    lines = [f"# 레이어 민감도 (A2m 기준, val {len(base[0])}장)", "",
             f"기준: A0 FP32 {fp[0].mean():.2%} / A2m {base[0].mean():.2%} (회복해야 할 손실 {gap*100:.2f}%p)", "",
             "각 행은 해당 묶음만 FP32 로 남기고 나머지 가중치 MatMul 은 A2m 과 똑같이 양자화한 결과다.", "",
             "| 구성 | FP32로 남긴 묶음 | 노드 수 | top-1 | A2m 대비 | 손실 회복률 | A2m만 정답 / 이 구성만 정답 | McNemar p (vs A2m) | FP32 일치율 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for code in codes:
        r = _acc(code)
        meta_p = C.RESULTS / "sensitivity" / f"{code}.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        acc = r[0].mean()
        mc = mcnemar(base[0], r[0])
        rec = (acc - base[0].mean()) / gap if gap > 0 else float("nan")
        lines.append(f"| {code} | {meta.get('spec', '?')} | {meta.get('n_nodes', '?')} | {acc:.2%} | "
                     f"{(acc - base[0].mean())*100:+.2f}%p | {rec:.0%} | {mc['a_only']} / {mc['b_only']} | "
                     f"{mc['p_value']:.2g} | {(r[1] == fp[1]).mean():.2%} |")
    lines += ["",
              "- 손실 회복률 = (이 구성 - A2m) / (A0 - A2m). 묶음끼리 효과가 겹칠 수 있어 합이 100% 가 되지 않는다",
              "- 표준오차 약 ±0.5%p. 짝지은 비교(McNemar)로 A2m 과의 차이가 우연인지 본다",
              "- 정확도만 쟀다. FP32 로 남긴 묶음은 속도를 지불하므로 최종 후보만 따로 속도를 잰다",
              "- 이 표는 ORT 정적 양자화에서의 민감도다. Hailo 에서 같은 층이 민감하다는 근거가 아니다"]
    out = C.RESULTS / "sensitivity.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[sens] 저장: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("groups")
    pn = sub.add_parser("names")
    pn.add_argument("spec")
    pn.add_argument("--out", required=True, help="노드 이름 목록 파일")
    pn.add_argument("--code", required=True)
    sub.add_parser("table")
    a = ap.parse_args()
    if a.cmd == "groups":
        cmd_groups()
    elif a.cmd == "table":
        cmd_table()
    else:
        names = select(load_rows(), a.spec)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(names) + "\n")
        (OUT_DIR / f"{a.code}.json").write_text(json.dumps(
            {"code": a.code, "spec": a.spec, "n_nodes": len(names), "nodes": names,
             "base": BASE_CODE, "timestamp": C.now()}, indent=2, ensure_ascii=False))
        print(f"[sens] {a.code}: {a.spec} -> {len(names)}개 노드 FP32 유지")


if __name__ == "__main__":
    main()
