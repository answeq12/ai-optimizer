"""E33 학습 데이터: COCO train2017 무작위 30,000장 목록 + 캡션 + 사진 받기.

  python -m src.da_data --n 2000     # 목록 앞 2,000장만 받기 (파일럿)
  python -m src.da_data              # 30,000장 전부

- 목록은 captions_train2017 의 사진 id 를 정렬해 시드 20261020 으로 섞은 앞 30,000장. 앞 N장 = 파일럿과 같은 사진
- 사진은 공식 주소에서 한 장씩 동시에 받는다(이어 받기 가능). val2017 은 건드리지 않는다
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import common as C
from . import da_common as D

URL = "http://images.cocodataset.org/train2017/{}"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
N_TOTAL = 30000


def captions() -> dict:
    if not D.CAP_TRAIN.exists():
        if not D.ANN_ZIP.exists():
            from .download_data import _download
            print("[da] 캡션 zip 다운로드 (약 241MB)", flush=True)
            _download(ANN_URL, D.ANN_ZIP)
        D.CAP_TRAIN.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(D.ANN_ZIP) as z:
            D.CAP_TRAIN.write_bytes(z.read("annotations/captions_train2017.json"))
    return json.loads(D.CAP_TRAIN.read_text())


def make_list(ann) -> list[str]:
    if D.TRAIN_LIST.exists():
        return D.read_list(D.TRAIN_LIST)
    imgs = sorted((im["id"], im["file_name"]) for im in ann["images"])
    perm = np.random.default_rng(D.SEED_LIST).permutation(len(imgs))
    names = [imgs[j][1] for j in perm[:N_TOTAL]]
    val = set(D.read_list(C.SPLITS / "coco_ret_select.txt")) | set(D.read_list(C.SPLITS / "coco_ret_confirm.txt"))
    assert not (set(names) & val), "train 목록에 평가 사진 이름이 섞임"
    D.TRAIN_LIST.write_text(f"# source=coco_train2017 seed={D.SEED_LIST} n={N_TOTAL}\n" + "\n".join(names) + "\n")
    return names


def make_full_list(ann) -> list[str]:
    """E37: 같은 시드 순서로 train2017 전체. 앞 30,000장이 E33 목록과 같은지 확인."""
    if D.FULL_LIST.exists():
        return D.read_list(D.FULL_LIST)
    imgs = sorted((im["id"], im["file_name"]) for im in ann["images"])
    perm = np.random.default_rng(D.SEED_LIST).permutation(len(imgs))
    names = [imgs[j][1] for j in perm]
    assert names[:N_TOTAL] == make_list(ann), "전체 목록 앞 30,000장이 E33 목록과 다름"
    val = set(D.read_list(C.SPLITS / "coco_ret_select.txt")) | set(D.read_list(C.SPLITS / "coco_ret_confirm.txt"))
    assert not (set(names) & val), "train 목록에 평가 사진 이름이 섞임"
    D.FULL_LIST.write_text(f"# source=coco_train2017 seed={D.SEED_LIST} n={len(names)} (앞 {N_TOTAL}장 = da_train_30k)\n" + "\n".join(names) + "\n")
    return names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_TOTAL)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    ann = captions()
    names = (make_full_list(ann) if a.n > N_TOTAL else make_list(ann))[:a.n]
    D.TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not ((D.TRAIN_DIR / n).exists() and (D.TRAIN_DIR / n).stat().st_size > 0)]
    print(f"[da] 목록 {len(names)}장, 받을 사진 {len(todo)}장", flush=True)

    def fetch(n):
        dst = D.TRAIN_DIR / n
        err = ""
        for _ in range(4):
            try:
                urllib.request.urlretrieve(URL.format(n), dst.with_suffix(".part"))
                dst.with_suffix(".part").rename(dst)
                return ""
            except Exception as e:
                err = f"{n}: {e}"
                time.sleep(2)
        return err

    t0, errs, nbytes = time.time(), [], 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, e in enumerate(ex.map(fetch, todo)):
            if e:
                errs.append(e)
            if (i + 1) % 1000 == 0:
                dt = time.time() - t0
                print(f"[da] {i + 1}/{len(todo)} ({dt:.0f}s, {(i + 1) / dt:.1f}장/s)", flush=True)
    dt = time.time() - t0
    nbytes = sum((D.TRAIN_DIR / n).stat().st_size for n in names if (D.TRAIN_DIR / n).exists())
    print(f"[da] 다운로드 {len(todo)}장 {dt:.0f}s, 목록 전체 {nbytes / 1e9:.2f} GB", flush=True)
    if errs:
        sys.exit(f"[da] 실패 {len(errs)}장 (다시 실행하면 이어 받음): {errs[:3]}")


if __name__ == "__main__":
    main()
