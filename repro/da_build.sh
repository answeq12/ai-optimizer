#!/usr/bin/env bash
# 최종 구성(DA5-SPL) 만들기: 적응 가중치 → FP32 ONNX(+ P1 게이트) + 적응 텍스트 임베딩 → 원본과 같은 SPL 처방 INT8
#   reproduce.sh / repro_quick.sh 가 부른다. COCO 캡션과 CIFAR-100 이 먼저 받아져 있어야 한다
#   가중치: repro/weights/da5_chosen.pt = {"img": 이미지 블록 8~11 + ln_post + proj, "txt": 텍스트 블록 8~11 + ln_final + text_projection}
#           (FP32, 학습 결과 그대로, sha256 확인). 크기 약 166 MB → GitHub Release 에서 받아 repro/weights/ 에 둔다
#   DA_TRAIN=1 이면 직접 학습한다(전체 재현 전용, CPU 약 15시간, COCO train2017 11.8만 장 + CC3M 10만 장 다운로드, 디스크 약 90 GB).
#     CPU 학습은 PC 마다 수치가 아주 조금 달라질 수 있어 결과도 약간 다를 수 있다
set -euo pipefail
S=$(python -m src.common store-dir); D="$S/models_da5"; mkdir -p "$D/imp"
W=repro/weights/da5_chosen.pt
if [ "${DA_TRAIN:-0}" = "1" ]; then
  echo "   적응 가중치 직접 학습 (DA_TRAIN=1)"
  N=118287
  python -m src.da_cache check
  python -m src.da_data --n $N
  python -m src.da_cache images --set train --n $N
  python -m src.da_cache images --set select
  python -m src.da_cache images --set cifarval
  python -m src.da_text check     # 텍스트 앞/뒤 나눔 확인 (학습 텍스트는 매 스텝 계산, 저장 안 함)
  python -m src.cc3m_eval prepare --train-shards 22 --val-shards 3
  python -m src.da_cache images --set cc3mval
  python -m src.da_cache images --set cc3m
  python - <<'PY'
# α 탐색에 필요한 select·CIFAR 텍스트 앞부분 저장 (작음)
from src import da_common as D
from src.da_text import _cache_texts, text_front
from src.coco_retrieval import captions_for
from src import common as C
m = D.load_clip(); _, _, sc = captions_for("select"); _cache_texts(m, sc, "tselect")
_cache_texts(m, [C.PROMPT_TEMPLATE.format(x) for x in C.cifar100_class_names()], "tcifar")
PY
  python -m src.da_text train-mix --n $N --tag e46
  python -m src.da_text sweep --tag e46 --ckpt da2_e46_ep2.pt --cc3m
  W="$S/models_da/da2_chosen_e46.pt"
elif [ -f "$W" ] && (cd repro/weights && sha256sum -c SHA256SUMS --quiet 2>/dev/null); then
  echo "   적응 가중치 sha256 일치 (제출본과 같은 파일)"
else
  echo "   ✘ $W 가 없거나 sha256 이 다릅니다. GitHub Release 의 da5_chosen.pt 를 repro/weights/ 에 두세요 (docs/06)"; exit 1
fi
export HALIO_MODELS_SUBDIR=models_da5
python -m src.da_text export --tag e46 --weights "$W"
CAL=splits/calib_coco_64_s0.txt
F="$D/imp/split_SPL-b567-k32.fp32.onnx"
[ -f "$F" ] || python -m src.split_outliers --blocks 5,6,7 --k 32 --calib-list $CAL --out "$F"
[ -f "$D/imp/SPL-b567-k32.onnx" ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel \
    --src "$F" --exclude-nodes-file "$F.hot_nodes.txt" --calib-list $CAL --out "$D/imp/SPL-b567-k32.onnx"
echo "   최종 구성: $D/imp/SPL-b567-k32.onnx (텍스트 상태 $D/text_state.pt)"
