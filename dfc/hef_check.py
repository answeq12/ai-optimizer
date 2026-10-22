"""보고서 3.3 'CPU 로 넘어가는 층이 있나' 확인. 컴파일된 HAR·HEF 만 읽는다 (새 컴파일 없음)

  python -m dfc.hef_check     -> logs/dfc/hef_check.md, logs/dfc/hef_check.json

확인 내용
1) 파싱은 translate_onnx_model(onnx, NET) 를 start/end 노드 지정 없이 호출했다 (dfc/smoke_test.py, dfc/split_hailo.py)
   → ONNX 전체 그래프가 Hailo 층으로 바뀌었고, 앞뒤를 잘라 CPU 에 남긴 부분이 없다는 뜻. 이 스크립트는 그 결과물을 다시 확인한다
2) HN 층 종류별 개수, 입력·출력 층과 모양 (입력 정규화가 네트워크 안에 있는지 포함)
3) hailortcli 가 있으면 parse-hef 출력 (네트워크 그룹·컨텍스트 수)
"""
from __future__ import annotations
import json, shutil, subprocess
from collections import Counter
from .smoke_test import HAILO_DIR, NET, LOG_DIR, hn_layers

TARGETS = [("H-official (공식)", ""), ("H-splf-b7-k32 (동결)", ".splf_b7_k32"), ("H-fc2b7 (참고)", ".coco64_fc2b7")]


def load_runner(har):
    from hailo_sdk_client import ClientRunner
    return ClientRunner(har=str(har), hw_arch="hailo10h")   # smoke_test.py 와 같은 호출


def main() -> None:
    L = ["# Hailo-10H 호환성 확인: CPU 로 넘어가는 층이 있나", "",
         "- 파싱 호출: `translate_onnx_model(onnx, NET)` — start/end 노드를 지정하지 않음 (ONNX 전체를 변환)", ""]
    res = {}
    for name, sfx in TARGETS:
        har = HAILO_DIR / f"{NET}{sfx}.compiled.har"; hef = HAILO_DIR / f"{NET}{sfx}.hailo10h.hef"
        L += [f"## {name}", "", f"- HAR: `{har.name}` {'있음' if har.exists() else '✘ 없음'} / HEF: `{hef.name}` {'있음 (%.1f MB)' % (hef.stat().st_size/1e6) if hef.exists() else '✘ 없음'}"]
        if not har.exists():
            L.append(""); continue
        try:
            layers = hn_layers(load_runner(har))
        except Exception as e:
            L += [f"- ✘ HAR 읽기 실패: {e}", ""]; continue
        types = Counter(v.get("type", "?") for v in layers.values())
        ins = {k: v.get("output_shapes", v.get("input_shapes")) for k, v in layers.items() if v.get("type") == "input_layer"}
        outs = {k: v.get("input_shapes") for k, v in layers.items() if v.get("type") == "output_layer"}
        suspicious = sorted({t for t in types if any(s in t.lower() for s in ("postprocess", "nms", "host", "external", "cpu"))})
        L += [f"- 층 {len(layers)}개, 종류 {len(types)}가지",
              f"- 입력 층: {ins}", f"- 출력 층: {outs}",
              f"- 호스트/후처리로 보이는 층 종류: {suspicious if suspicious else '없음'}",
              f"- 정규화 층 존재(입력 정규화가 칩 안): {'예' if any('normalization' in t for t in types) else '확인 필요'}",
              "", "| 층 종류 | 개수 |", "|---|---|"] + [f"| {t} | {c} |" for t, c in types.most_common()]
        info = {"layers": len(layers), "types": dict(types), "inputs": ins, "outputs": outs, "suspicious": suspicious}
        cli = shutil.which("hailortcli")
        if cli and hef.exists():
            try:
                o = subprocess.run([cli, "parse-hef", str(hef)], capture_output=True, text=True, timeout=120)
                L += ["", "hailortcli parse-hef:", "```", (o.stdout + o.stderr).strip()[:3000], "```"]
                info["parse_hef"] = o.stdout
            except Exception as e:
                L.append(f"- hailortcli 실행 실패: {e}")
        else:
            L.append("- hailortcli 없음 (HailoRT 미설치) → HEF 구조는 보드에서 본선 때 확인")
        res[name] = info; L.append("")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "hef_check.md").write_text("\n".join(L) + "\n")
    (LOG_DIR / "hef_check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main()
