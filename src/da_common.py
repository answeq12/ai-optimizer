"""도메인 적응(E33) 공용: 경로, 모델 앞/뒤 나누기, 특징 저장 형식.

블록 0~7(앞)은 얼리고 블록 8~11 + ln_post + proj(뒤)만 학습한다.
앞쪽 출력(블록 7 출력, [197, 768])을 fp16 으로 저장해 두고 학습·α 탐색은 뒤쪽만 돌린다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import common as C

SPLIT_AT = 8                                   # 블록 0~7 얼림, 8~11 학습
DA_DIR = C.STORE / "models_da"                 # 학습 가중치·변환 모델
FEAT_DIR = Path(os.environ.get("HALIO_DA_FEAT", str(C.STORE / "da_feat")))   # 블록 7 출력 저장
TRAIN_LIST = C.SPLITS / "da_train_30k.txt"
FULL_LIST = C.SPLITS / "da_train_full.txt"   # E37: train2017 전체(같은 시드 순서, 앞 30k = TRAIN_LIST)
TRAIN_DIR = C.DATA / "coco" / "train2017"
ANN_ZIP = C.DATA / "coco" / "annotations_trainval2017.zip"
CAP_TRAIN = C.DATA / "coco" / "annotations" / "captions_train2017.json"
SEED_LIST = 20261020


def load_clip():
    import open_clip
    m = open_clip.create_model(C.MODEL_NAME, pretrained=C.PRETRAINED, force_quick_gelu=True,
                               cache_dir=str(C.CACHE)).eval()
    v = m.visual
    assert getattr(v, "attn_pool", None) is None, "attn_pool 이 있는 모델은 지원하지 않음"
    assert not getattr(v, "final_ln_after_pool", False), "final_ln_after_pool=True 는 지원하지 않음"
    assert getattr(v, "pool_type", "tok") == "tok", f"pool_type={getattr(v, 'pool_type', None)}"
    return m


def _run_blocks(v, x, blocks):
    bf = getattr(v.transformer, "batch_first", True)
    if not bf:
        x = x.transpose(0, 1)
    for b in blocks:
        x = b(x)
    if not bf:
        x = x.transpose(0, 1)
    return x


def front(v, x):
    """[N,3,224,224] → 블록 7 출력 [N,197,768] (open_clip VisionTransformer.forward 앞부분과 같게)."""
    import torch
    x = v.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype)
    x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
    x = v.patch_dropout(x)
    x = v.ln_pre(x)
    return _run_blocks(v, x, v.transformer.resblocks[:SPLIT_AT])


def make_tail(v):
    import torch.nn as nn

    class Tail(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(list(v.transformer.resblocks[SPLIT_AT:]))
            self.ln_post = v.ln_post
            self.proj = v.proj
            self._v = [v]                      # 리스트로 감싸 하위 모듈 등록 방지

        def forward(self, x):
            x = _run_blocks(self._v[0], x, self.blocks)
            x = self.ln_post(x)
            return x[:, 0] @ self.proj

    return Tail()


def tail_state(tail) -> dict:
    return {k: p.detach().clone().float() for k, p in tail.state_dict().items()}


def feat_path(name: str) -> Path:
    return FEAT_DIR / f"{name}.f16"


def write_meta(name: str, n: int, extra: dict) -> None:
    (FEAT_DIR / f"{name}.json").write_text(json.dumps({"n": n, "shape": [n, 197, 768], "dtype": "float16", **extra},
                                                      indent=2, ensure_ascii=False))


def open_feat(name: str, mode="r"):
    meta = json.loads((FEAT_DIR / f"{name}.json").read_text())
    return np.memmap(feat_path(name), dtype=np.float16, mode=mode, shape=tuple(meta["shape"])), meta


def train_list(n: int) -> Path:
    """n ≤ 30,000 이면 E33 목록, 넘으면 전체 목록 (앞 30,000장은 같다)."""
    return FULL_LIST if n > 30000 else TRAIN_LIST


def read_list(p: Path) -> list[str]:
    return [ln.strip() for ln in Path(p).read_text().split("\n")[1:] if ln.strip()]
