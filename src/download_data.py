"""데이터셋 다운로드. 원본은 저장소에 넣지 않고 이 스크립트로 받는다.

  python -m src.download_data            # CIFAR-100 + COCO val2017
  python -m src.download_data --skip-coco

- CIFAR-100 (약 170MB): torchvision 이 받아서 md5 를 검증한다. train 과 test 가 함께 들어 있다.
- COCO val2017 (약 780MB, 5,000장): 캘리브레이션 원본 풀. 라벨은 받지 않는다.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile

from . import common as C

COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_COUNT = 5000


def get_cifar() -> None:
    if (C.CIFAR_DIR / "train").exists() and (C.CIFAR_DIR / "test").exists():
        print("[cifar] 이미 있음:", C.CIFAR_DIR)
        return
    from torchvision.datasets import CIFAR100
    CIFAR100(root=str(C.DATA), train=True, download=True)   # md5 검증 포함
    print("[cifar] 완료:", C.CIFAR_DIR)


def _download(url: str, dst) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        if done and r.status != 206:      # 서버가 이어받기를 지원하지 않으면 처음부터
            done = 0
        total = done + int(r.headers.get("Content-Length") or 0)
        with open(tmp, "ab" if done else "wb") as fh:
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                fh.write(buf)
                done += len(buf)
                if total:
                    print(f"\r[coco] {done/1e6:7.1f} / {total/1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dst)


def get_coco() -> None:
    n = len(list(C.COCO_DIR.glob("*.jpg"))) if C.COCO_DIR.exists() else 0
    if n == COCO_COUNT:
        print("[coco] 이미 있음:", C.COCO_DIR)
        return
    zpath = C.DATA / "coco" / "val2017.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    if not zpath.exists():
        _download(COCO_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(zpath.parent)
    n = len(list(C.COCO_DIR.glob("*.jpg")))
    if n != COCO_COUNT:
        sys.exit(f"[coco] 이미지 수가 {n}장. {COCO_COUNT}장이어야 한다. zip 을 지우고 다시 받으세요.")
    print("[coco] 완료:", C.COCO_DIR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-coco", action="store_true")
    args = ap.parse_args()
    C.ensure_dirs()
    free_gb = shutil.disk_usage(C.DATA).free / 1e9
    if free_gb < 5:
        sys.exit(f"디스크 여유 {free_gb:.1f}GB. 최소 5GB 가 필요합니다.")
    get_cifar()
    if not args.skip_coco:
        get_coco()
    rec = {
        "cifar100": {"url": "https://www.cs.toronto.edu/~kriz/cifar.html",
                     "verified_by": "torchvision md5"},
        "coco_val2017": {"url": COCO_URL, "images": COCO_COUNT, "labels": "받지 않음"},
        "downloaded_at": C.now(),
    }
    (C.DATA / "SOURCES.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
