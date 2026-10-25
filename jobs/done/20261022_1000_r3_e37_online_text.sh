# E37 재투입 3: C 드라이브 부족(WSL 가상디스크) 대응 — 미완성 이미지 저장분 지우고, 텍스트 앞부분은 저장하지 않고 매 스텝 계산(--online-text). 나머지는 사전 등록과 같음
set -eo pipefail
source ~/venvs/halio/bin/activate
export HALIO_DA_FEAT=$(cat logs/da_feat_dir.txt)
N=118287
rm -f $HALIO_DA_FEAT/train_${N}.f16.part
df -h $HOME /mnt/c | cat
CFREE=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9); echo "C 여유 ${CFREE}GB"
[ "$CFREE" -ge 20 ] || { echo "C 여유 20GB 미만 → 중단"; exit 1; }
# 매 스텝 계산 = 저장분과 같은지 확인 (앞 30k 캡션 저장분과 16개 비교)
python - <<'PY'
import numpy as np, torch
from src import da_common as D
from src.da_text import text_front, tok, open_t, train_caps
texts = train_caps(30000); TF, TE = open_t("ttrain_30000"); m = D.load_clip()
idx = np.sort(np.random.default_rng(1).choice(len(texts), 16, replace=False))
with torch.no_grad():
    f, e = text_front(m, tok()([texts[i] for i in idx]))
err = float(np.abs(f.half().float().numpy() - np.asarray(TF[idx], dtype=np.float32)).max())
print(f"[e37] 매 스텝 계산 vs 저장분: 최대 차 {err:.2e}, eot 동일 {np.array_equal(e.numpy(), TE[idx])}")
assert err < 0.5 and np.array_equal(e.numpy(), TE[idx])
PY
python -m src.da_cache images --set train --n $N
CFREE=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9); echo "이미지 저장 후 C 여유 ${CFREE}GB"
timeout -k 60 8h python -m src.da_text train --n $N --tag e37 --online-text
python -m src.da_text sweep --tag e37 --ckpt da2_e37_ep2.pt
[ -f /mnt/d/halio_data/models_da/da2_chosen_e37.pt ] || { echo "선택 없음 → 종료"; exit 0; }
export HALIO_MODELS_SUBDIR=models_da3; M=/mnt/d/halio_data/models_da3; mkdir -p $M/imp
python -m src.da_text export --tag e37
CAL=splits/calib_coco_64_s0.txt; F=$M/imp/split_SPL-b567-k32.fp32.onnx
[ -f $F ] || python -m src.split_outliers --blocks 5,6,7 --k 32 --calib-list $CAL --out $F
[ -f $M/imp/SPL-b567-k32.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --src $F --exclude-nodes-file $F.hot_nodes.txt --calib-list $CAL --out $M/imp/SPL-b567-k32.onnx
cmp -s /mnt/d/halio_data/models/imp/split_SPL-b567-k32.fp32.onnx.hot_nodes.txt $F.hot_nodes.txt && echo "보호 노드 원본과 동일" || echo "보호 노드 다름"
python -m src.evaluate --model $M/clip_vitb16_image_fp32.onnx --code DA3-A0 --quant-mode fp32 --eval-set val | tail -1
python -m src.evaluate --model $M/imp/SPL-b567-k32.onnx --code DA3-SPL --quant-mode static --eval-set val | tail -1
python -m src.coco_retrieval eval --model $M/clip_vitb16_image_fp32.onnx --code DA3-A0 --split select | tail -1
python -m src.coco_retrieval eval --model $M/imp/SPL-b567-k32.onnx --code DA3-SPL --split select | tail -1
python -m src.coco_retrieval table --ref A0 --codes A0 DA2-A0 DA2-SPL DA3-A0 DA3-SPL --split select --margin 2.0 --tag e37
python -m src.coco_retrieval table --ref DA2-SPL --codes DA2-SPL DA3-A0 DA3-SPL --split select --margin 2.0 --tag e37_vs_da2
