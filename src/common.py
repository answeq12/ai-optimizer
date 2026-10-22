"""전 조건이 공유하는 단일 소스: 경로, 전처리, 데이터 로더, 환경 기록, CSV 스키마.

전처리와 데이터 로딩은 여기 외에 중복 구현하지 않는다. 한 곳이라도 어긋나면 비교가 무효다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _store_dir() -> Path:
    """용량이 큰 파일(데이터·모델·가중치 캐시)을 둘 곳.

    환경변수 HALIO_DATA_DIR 가 있으면 그곳, 없고 D 드라이브(/mnt/d)가 보이면 /mnt/d/halio_data,
    둘 다 아니면 프로젝트 폴더. 코드·결과·로그·split 은 항상 프로젝트 폴더에 둔다.
    """
    env = os.environ.get("HALIO_DATA_DIR")
    if env:
        return Path(env).expanduser()
    if Path("/mnt/d").is_dir():
        return Path("/mnt/d/halio_data")
    return ROOT


STORE = _store_dir()
DATA = STORE / "data"
MODELS = STORE / os.environ.get("HALIO_MODELS_SUBDIR", "models")   # 다른 모델 실험(사후, 예: ViT-B-32)만 바꾼다
CACHE = STORE / "cache"
RESULTS = ROOT / "results"
SPLITS = ROOT / "splits"
LOGS = ROOT / "logs"

# ---- 고정 프로토콜 값 (docs/PROTOCOL.md 와 일치해야 한다) -------------------
MODEL_NAME = os.environ.get("HALIO_MODEL", "ViT-B-16")   # open_clip 모델 이름. 기본 ViT-B-16 (일반화 확인 실험만 바꾼다)
PRETRAINED = "openai"            # OpenAI 공개 가중치 (MIT)
IMAGE_SIZE = int(os.environ.get("HALIO_IMAGE_SIZE", "224"))   # 기본 224. 해상도 축소 실험(src/export_res.py)만 바꾼다
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
PROMPT_TEMPLATE = "a photo of a {}"
OPSET = 17
WARMUP = 20
RUNS = 300
DEFAULT_THREADS = 4              # 측정 스레드 수. 팀 결정 전 기본값(라즈베리파이 5의 4코어)
VAL_PER_CLASS = 100              # 검증셋 클래스당 장수 (100클래스 x 100 = 10,000장). 2026-10-05 결정
SPLIT_SEED = 20261005
CALIB_SIZE = 64
CALIB_SEEDS = (0, 1, 2)

RAW_ONNX = MODELS / "clip_vitb16_image_fp32.raw.onnx"
FP32_ONNX = MODELS / "clip_vitb16_image_fp32.onnx"
TEXT_EMB = MODELS / "text_emb_cifar100.npy"
CLASS_NAMES = MODELS / "cifar100_classes.json"

CSV_FIELDS = [
    "run_id", "config_code", "backend", "model_name", "model_path",
    "quant_mode", "calib_method", "per_channel", "reduce_range", "excluded_nodes",
    "calib_source", "calib_size", "calib_seed", "eval_set", "eval_n",
    "top1_acc", "recall_at_1", "recall_at_5",
    "latency_p50_ms", "latency_p95_ms", "n_threads",
    "max_rss_mb", "model_size_mb", "quantized_op_ratio",
    "env_hash", "timestamp", "repeat_idx",
]


def ensure_dirs() -> None:
    for d in (DATA, MODELS, CACHE, RESULTS, SPLITS, LOGS):
        d.mkdir(parents=True, exist_ok=True)


# ---- 전처리 -----------------------------------------------------------------
def preprocess_uint8(img: Image.Image, size: int | None = None) -> np.ndarray:
    """PIL 이미지 -> uint8 [224, 224, 3] (RGB, 0~255). 리사이즈와 자르기까지만 한다.

    짧은 변을 224로 bicubic 리사이즈한 뒤 중앙 224x224 를 자른다(CLIP 공식 전처리와 동일).
    CIFAR-100(32x32)은 정사각형이라 224x224 bicubic 업샘플만 일어난다.
    정규화를 네트워크 안에서 하는 백엔드(Hailo, normalize_in_net)는 이 값을 그대로 받는다.
    """
    S = size or IMAGE_SIZE
    img = img.convert("RGB")
    w, h = img.size
    s = S / min(w, h)
    nw, nh = max(S, round(w * s)), max(S, round(h * s))
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - S) // 2, (nh - S) // 2
    img = img.crop((left, top, left + S, top + S))
    return np.asarray(img, dtype=np.uint8)


def preprocess(img: Image.Image, size: int | None = None) -> np.ndarray:
    """PIL 이미지 -> float32 [3, S, S] (기본 S=224), CLIP 정규화 적용 (ORT 경로의 입력)."""
    x = preprocess_uint8(img, size).astype(np.float32) / 255.0
    x = (x - np.asarray(CLIP_MEAN, dtype=np.float32)) / np.asarray(CLIP_STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# ---- CIFAR-100 --------------------------------------------------------------
CIFAR_DIR = DATA / "cifar-100-python"


def load_cifar100(split: str):
    """split: 'train'(50,000) 또는 'test'(10,000). (uint8 [N,32,32,3], int64 [N]) 반환."""
    f = CIFAR_DIR / split
    if not f.exists():
        raise FileNotFoundError(f"{f} 없음. 먼저 `python -m src.download_data` 를 실행하세요.")
    with open(f, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    data = np.asarray(d["data"], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(d["fine_labels"], dtype=np.int64)
    return data, labels


def cifar100_class_names() -> list[str]:
    with open(CIFAR_DIR / "meta", "rb") as fh:
        meta = pickle.load(fh, encoding="latin1")
    return [n.replace("_", " ") for n in meta["fine_label_names"]]


def read_index_list(path: Path) -> np.ndarray:
    return np.asarray([int(x) for x in Path(path).read_text().split()], dtype=np.int64)


def load_eval_set(name: str):
    """평가셋 이름 -> (images uint8 [N,32,32,3], labels [N]).

    'val'  : CIFAR-100 train 에서 분리한 검증셋 (설정 선택용)
    'test' : CIFAR-100 test 10,000장. 최종 구성 확정 전에는 열지 않는다.
    """
    if name == "val":
        data, labels = load_cifar100("train")
        idx = read_index_list(SPLITS / "val.txt")
        return data[idx], labels[idx]
    if name == "test":
        return load_cifar100("test")
    raise ValueError(f"알 수 없는 평가셋: {name}")


# ---- 캘리브레이션 입력 -------------------------------------------------------
COCO_DIR = DATA / "coco" / "val2017"


def _guard_coco_calib(items: list[str], list_path) -> None:
    """COCO 캘리브레이션 사진이 검색 평가(select/confirm)와 겹치면 멈춘다.
    새 캘리브레이션 사진은 splits/coco_ret_exclude.txt 안의 사진이나 COCO train2017 에서만 뽑는다."""
    for name in ("select", "confirm"):
        sp = SPLITS / f"coco_ret_{name}.txt"
        if sp.exists():
            ev = {ln.strip() for ln in sp.read_text().split("\n")[1:] if ln.strip()}
            bad = sorted(set(items) & ev)
            if bad:
                raise SystemExit(f"{list_path}: 캘리브레이션 사진 {len(bad)}장이 COCO-{name} 평가 사진과 겹칩니다 (예: {bad[:3]}). "
                                 "캘리브레이션은 coco_ret_exclude.txt 안의 사진이나 COCO train2017 에서만 뽑으세요.")


def load_calib_images(list_path: Path, measure_only: bool = False) -> np.ndarray:
    """캘리브레이션 목록 파일 -> float32 [N,3,224,224].

    목록 첫 줄은 '# source=coco' 또는 '# source=cifar_train'. 이후 한 줄에 항목 하나.
    """
    lines = Path(list_path).read_text().split("\n")
    header = lines[0].strip()
    items = [ln.strip() for ln in lines[1:] if ln.strip()]
    if header == "# source=coco":
        if not measure_only:   # measure_only: 활성값 통계 측정 전용(양자화 범위 추정에 쓰지 않음)
            _guard_coco_calib(items, list_path)
        return np.stack([preprocess(Image.open(COCO_DIR / n)) for n in items])
    if header == "# source=coco_train":
        return np.stack([preprocess(Image.open(DATA / "coco" / "train2017" / n)) for n in items])
    if header == "# source=cifar_train":
        data, _ = load_cifar100("train")
        return np.stack([preprocess(Image.fromarray(data[int(i)])) for i in items])
    raise ValueError(f"{list_path}: 첫 줄이 '# source=...' 가 아님")


def load_calib_images_uint8(list_path: Path) -> np.ndarray:
    """캘리브레이션 목록 파일 -> uint8 [N,224,224,3]. 정규화를 네트워크 안에서 하는 백엔드용."""
    lines = Path(list_path).read_text().split("\n")
    header = lines[0].strip()
    items = [ln.strip() for ln in lines[1:] if ln.strip()]
    if header == "# source=coco":
        _guard_coco_calib(items, list_path)
        return np.stack([preprocess_uint8(Image.open(COCO_DIR / n)) for n in items])
    if header == "# source=coco_train":          # (10/21 E41) COCO train2017 사진. 평가(val2017)와 겹칠 수 없음
        return np.stack([preprocess_uint8(Image.open(DATA / "coco" / "train2017" / n)) for n in items])
    if header == "# source=cifar_train":
        data, _ = load_cifar100("train")
        return np.stack([preprocess_uint8(Image.fromarray(data[int(i)])) for i in items])
    raise ValueError(f"{list_path}: 첫 줄이 '# source=...' 가 아님")


def calib_source_of(list_path: Path) -> str:
    return Path(list_path).read_text().split("\n", 1)[0].replace("# source=", "").strip()


# ---- 환경 기록 ---------------------------------------------------------------
def _cpu_info() -> dict:
    info = {"model": platform.processor() or "", "flags": []}
    try:
        txt = Path("/proc/cpuinfo").read_text()
        for line in txt.split("\n"):
            if line.startswith("model name") and not info.get("model_name"):
                info["model_name"] = line.split(":", 1)[1].strip()
            if line.startswith("flags") and not info["flags"]:
                fl = set(line.split(":", 1)[1].split())
                info["flags"] = sorted(f for f in fl if f in {
                    "avx2", "avx512f", "avx512_vnni", "avx_vnni", "fma", "sse4_2"})
    except OSError:
        pass
    info["logical_cpus"] = os.cpu_count()
    return info


def _mem_total_gb():
    try:
        for line in Path("/proc/meminfo").read_text().split("\n"):
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        return None


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def env_info() -> dict:
    vers = {}
    for mod in ("numpy", "onnx", "onnxruntime", "torch", "open_clip", "PIL"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "?")
        except Exception:
            vers[mod] = None
    rel = platform.release()
    return {
        "os": platform.platform(),
        "kernel": rel,
        "is_wsl": "microsoft" in rel.lower(),
        "python": sys.version.split()[0],
        "cpu": _cpu_info(),
        "mem_total_gb": _mem_total_gb(),
        "versions": vers,
        "git_commit": _git_commit(),
        "store_dir": str(STORE),
    }


def env_hash(info: dict | None = None) -> str:
    info = dict(info or env_info())
    info.pop("git_commit", None)
    info.pop("store_dir", None)
    return hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:10]


def dump_env() -> str:
    """환경 정보를 results/env_<hash>.json 으로 저장하고 해시를 돌려준다."""
    ensure_dirs()
    info = env_info()
    h = env_hash(info)
    p = RESULTS / f"env_{h}.json"
    if not p.exists():
        p.write_text(json.dumps(info, indent=2, ensure_ascii=False))
    return h


# ---- 결과 CSV ----------------------------------------------------------------
def append_row(csv_path: Path, row: dict) -> None:
    """고정 스키마로 한 줄 추가. 스키마에 없는 키는 오류로 처리한다."""
    unknown = set(row) - set(CSV_FIELDS)
    if unknown:
        raise KeyError(f"CSV 스키마에 없는 열: {sorted(unknown)}")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def size_mb(path: Path) -> float:
    return round(Path(path).stat().st_size / 1e6, 2)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":      # run_pilot.sh 가 경로를 물어볼 때 쓴다
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["models-dir", "data-dir", "store-dir"])
    a = ap.parse_args()
    print({"models-dir": MODELS, "data-dir": DATA, "store-dir": STORE}[a.what])
