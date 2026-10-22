"""빠른 재현용 데이터 받기 (repro_quick.sh). 전체 COCO val2017(815 MB) 대신 필요한 사진만 받는다.

  python -m src.quick_data

- CIFAR-100 (약 170 MB, torchvision 이 md5 검증): FP32 export 의 PyTorch 일치 검사와 CIFAR test 앞 2,000장에 쓴다
- COCO val2017 사진 564장(confirm 앞 500장 + 캘리브 64장)만 공식 주소에서 한 장씩 받는다 (약 90 MB)
- 캡션은 저장소의 repro/quick/captions_quick.json (COCO captions_val2017 중 위 사진 부분, CC BY 4.0 주석)
"""
from __future__ import annotations
import sys, urllib.request
from . import common as C
from .download_data import get_cifar

URL = "http://images.cocodataset.org/val2017/{}"
LIST = C.ROOT / "repro" / "quick" / "coco_files.txt"


def main() -> None:
    C.ensure_dirs()
    get_cifar()
    C.COCO_DIR.mkdir(parents=True, exist_ok=True)
    names = [l.strip() for l in LIST.read_text().split("\n") if l.strip() and not l.startswith("#")]
    todo = [n for n in names if not ((C.COCO_DIR / n).exists() and (C.COCO_DIR / n).stat().st_size > 0)]

    def fetch(n: str) -> str:
        dst = C.COCO_DIR / n
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(URL.format(n), dst.with_suffix(".part"))
                dst.with_suffix(".part").rename(dst)
                return ""
            except Exception as e:
                err = f"{n}: {e}"
        return err

    from concurrent.futures import ThreadPoolExecutor   # 8개씩 동시에 받는다 (한 장씩이면 약 15분)
    errs = []
    with ThreadPoolExecutor(8) as ex:
        for i, e in enumerate(ex.map(fetch, todo)):
            if e:
                errs.append(e)
            if (i + 1) % 100 == 0:
                print(f"[quick] {i + 1}/{len(todo)}", flush=True)
    if errs:
        sys.exit(f"[quick] 다운로드 실패 {len(errs)}장 (다시 실행하면 이어 받음): {errs[:3]}")
    got = len(todo)
    print(f"[quick] COCO 사진 {len(names)}장 준비 (새로 받음 {got}장): {C.COCO_DIR}")


if __name__ == "__main__":
    main()
