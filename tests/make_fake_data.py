"""다운로드 없이 코드 경로만 검증하기 위한 가짜 데이터 생성기. 결과 수치는 무의미하다.

  python -m tests.make_fake_data

data/ 에 CIFAR-100 형식 pickle(train 3,000장 / test 300장)과 COCO 형식 jpg 200장을 무작위로 만든다.
이미 진짜 데이터가 있으면 덮어쓰지 않고 중단한다.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
from PIL import Image

from src import common as C


def main() -> None:
    if C.CIFAR_DIR.exists() or C.COCO_DIR.exists():
        sys.exit("data/ 에 이미 데이터가 있습니다. 가짜 데이터로 덮어쓰지 않습니다.")
    rng = np.random.default_rng(0)
    C.CIFAR_DIR.mkdir(parents=True)
    # train 은 검증셋(클래스당 C.VAL_PER_CLASS)을 떼고도 캘리브레이션 풀이 남도록 +10장
    for split, per in (("train", C.VAL_PER_CLASS + 10), ("test", 3)):
        labels = np.repeat(np.arange(100), per)
        rng.shuffle(labels)
        data = rng.integers(0, 256, size=(len(labels), 3072), dtype=np.uint8)
        with open(C.CIFAR_DIR / split, "wb") as fh:
            pickle.dump({"data": data, "fine_labels": labels.tolist()}, fh)
    with open(C.CIFAR_DIR / "meta", "wb") as fh:
        pickle.dump({"fine_label_names": [f"class_{i}" for i in range(100)]}, fh)
    C.COCO_DIR.mkdir(parents=True)
    for i in range(200):
        w, h = int(rng.integers(240, 640)), int(rng.integers(240, 480))
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(arr).save(C.COCO_DIR / f"{i:012d}.jpg", quality=80)
    (C.DATA / "FAKE_DATA").write_text("tests/make_fake_data.py 가 만든 가짜 데이터. 실험에 쓰지 말 것.\n")
    print("가짜 데이터 생성 완료:", C.DATA)


if __name__ == "__main__":
    main()
