"""검증셋 분리와 캘리브레이션 목록 생성. split 을 만드는 유일한 진입점.

  python -m src.split_val            # 생성 (이미 있으면 같은지 대조만 한다)
  python -m src.split_val --verify   # 대조만

산출물 (splits/)
  val.txt                      CIFAR-100 train 인덱스. 클래스당 VAL_PER_CLASS 장
  calib_coco_64_s{0,1,2}.txt   COCO val2017 파일명 64개씩 (A 조건)
  calib_cifar_64_s{0,1,2}.txt  CIFAR-100 train 인덱스 64개씩 (B 조건, 검증셋과 비중복)

CIFAR-100 test 10,000장은 여기서 읽지 않는다.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import common as C


def build() -> dict[str, str]:
    """모든 split 파일의 내용을 결정론적으로 만든다. {파일명: 내용}."""
    out: dict[str, str] = {}
    _, labels = C.load_cifar100("train")
    rng = np.random.default_rng(C.SPLIT_SEED)

    val = []
    for c in range(100):
        idx = np.flatnonzero(labels == c)
        val.extend(rng.choice(idx, size=C.VAL_PER_CLASS, replace=False).tolist())
    val = sorted(val)
    out["val.txt"] = "\n".join(map(str, val)) + "\n"

    pool = np.setdiff1d(np.arange(len(labels)), np.asarray(val))   # 검증셋을 뺀 train
    coco = sorted(p.name for p in C.COCO_DIR.glob("*.jpg")) if C.COCO_DIR.exists() else []
    for s in C.CALIB_SEEDS:
        r = np.random.default_rng(C.SPLIT_SEED + 1000 + s)
        pick = sorted(r.choice(pool, size=C.CALIB_SIZE, replace=False).tolist())
        out[f"calib_cifar_{C.CALIB_SIZE}_s{s}.txt"] = \
            "# source=cifar_train\n" + "\n".join(map(str, pick)) + "\n"
        if coco:
            r = np.random.default_rng(C.SPLIT_SEED + 2000 + s)
            pick = sorted(r.choice(coco, size=C.CALIB_SIZE, replace=False).tolist())
            out[f"calib_coco_{C.CALIB_SIZE}_s{s}.txt"] = \
                "# source=coco\n" + "\n".join(pick) + "\n"
    return out


def check_disjoint() -> None:
    """검증셋과 CIFAR 캘리브레이션 목록의 교집합이 0인지, 클래스 균형이 맞는지 확인."""
    _, labels = C.load_cifar100("train")
    val = C.read_index_list(C.SPLITS / "val.txt")
    assert len(val) == len(set(val.tolist())) == 100 * C.VAL_PER_CLASS, "검증셋 크기/중복 오류"
    counts = np.bincount(labels[val], minlength=100)
    assert (counts == C.VAL_PER_CLASS).all(), "검증셋 클래스 균형 오류"
    vs = set(val.tolist())
    for p in sorted(C.SPLITS.glob("calib_cifar_*.txt")):
        items = {int(x) for x in p.read_text().split("\n")[1:] if x.strip()}
        inter = vs & items
        assert not inter, f"{p.name}: 검증셋과 {len(inter)}장 겹침"
    print(f"[split] 검증셋 {len(val)}장 (클래스당 {C.VAL_PER_CLASS}), "
          f"CIFAR 캘리브레이션 목록과의 교집합 0 확인")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    C.ensure_dirs()
    files = build()
    if not any(k.startswith("calib_coco") for k in files):
        print("[split] 경고: COCO val2017 이 없어 COCO 캘리브레이션 목록을 만들지 않았습니다.")
    for name, text in files.items():
        p = C.SPLITS / name
        if p.exists():
            if p.read_text() != text:
                sys.exit(f"[split] {name} 이 재생성 결과와 다릅니다. "
                         "split 은 한 번 정하면 바꾸지 않습니다. 의도한 변경이면 파일을 지우고 다시 실행하세요.")
        elif args.verify:
            sys.exit(f"[split] {name} 없음")
        else:
            p.write_text(text)
            print("[split] 생성:", p.name)
    check_disjoint()
    print("[split] 재실행 시 동일한 목록이 재현됨을 확인")


if __name__ == "__main__":
    main()
