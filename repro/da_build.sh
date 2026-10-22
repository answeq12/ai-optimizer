#!/usr/bin/env bash
# 최종 구성(DA-SPL-b567-k32) 만들기: 도메인 적응 가중치 → FP32 ONNX + P1 게이트 → 원본과 같은 SPL 처방 INT8
#   reproduce.sh / repro_quick.sh 가 부른다. 원본 FP32 ONNX 와 COCO 캡션 텍스트 임베딩이 먼저 있어야 한다
#   가중치: repro/weights/da_tail_chosen.pt (블록 8~11 + ln_post + proj, FP32, 학습 결과 그대로). sha256 확인
#   DA_TRAIN=1 이면 가중치를 받지 않고 직접 학습해 만든다(전체 재현 전용, 약 3시간, COCO train2017 3만 장 다운로드).
#     CPU 학습은 PC 마다 수치가 아주 조금 달라질 수 있어 결과도 약간 다를 수 있다
set -euo pipefail
S=$(python -m src.common store-dir); D="$S/models_da"; mkdir -p "$D/imp"
if [ ! -f "$D/da_tail_chosen.pt" ]; then
  if [ "${DA_TRAIN:-0}" = "1" ]; then
    echo "   적응 가중치 직접 학습 (DA_TRAIN=1)"
    python -m src.da_cache check
    python -m src.da_data
    python -m src.da_cache images --set train --n 30000
    python -m src.da_cache images --set select
    python -m src.da_cache images --set cifarval
    python -m src.da_cache text --n 30000
    python -m src.da_train --n 30000
    python -m src.da_sweep --ckpt da_tail_ep2.pt
  else
    cp repro/weights/da_tail_chosen.pt repro/weights/da_tail_chosen.json "$D/"
  fi
fi
if (cd "$D" && sha256sum -c "$OLDPWD/repro/weights/SHA256SUMS" --quiet 2>/dev/null); then
  echo "   적응 가중치 sha256 일치 (제출본과 같은 파일)"
else
  echo "   적응 가중치 sha256 다름 (DA_TRAIN=1 로 새로 학습했다면 정상)"
fi
export HALIO_MODELS_SUBDIR=models_da
[ -f "$D/clip_vitb16_image_fp32.onnx" ] || python -m src.da_export
python -m src.da_export --copy-text-only
CAL=splits/calib_coco_64_s0.txt
F="$D/imp/split_SPL-b567-k32.fp32.onnx"
[ -f "$F" ] || python -m src.split_outliers --blocks 5,6,7 --k 32 --calib-list $CAL --out "$F"
[ -f "$D/imp/SPL-b567-k32.onnx" ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel \
    --src "$F" --exclude-nodes-file "$F.hot_nodes.txt" --calib-list $CAL --out "$D/imp/SPL-b567-k32.onnx"
echo "   최종 구성: $D/imp/SPL-b567-k32.onnx"
