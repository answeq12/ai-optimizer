# 최종 판정 1회 (10/17 15:30 투입). 구성 동결: 10/17 15:22 사용자 지시 (PROTOCOL 선택 규칙 1~5 적용 결과)
#   ORT  후보 SPL-b567-k32 (정적 INT8 + b5~7 fc2 이상치 32채널 FP32) / 기준선 A0
#   Hailo 후보 H-splf-b7-k32 (fc1+GELU+fc2 이상치 32채널만 a16_w16) / 기준선 H-FP, H-official
# COCO-confirm(2406장) 텍스트→이미지 검색 + CIFAR-100 test 10k. 결과가 나빠도 후보를 바꾸지 않는다
set -e
source ~/venvs/halio/bin/activate
M=$(python -m src.common models-dir)
echo "== ORT $(date '+%F %T')"
python -m src.coco_retrieval text
python -m src.coco_retrieval eval --model $M/clip_vitb16_image_fp32.onnx --code A0 --split confirm --i-know-final
python -m src.coco_retrieval eval --model $M/imp/SPL-b567-k32.onnx --code SPL-b567-k32 --split confirm --i-know-final
python -m src.evaluate --model $M/clip_vitb16_image_fp32.onnx --code A0 --quant-mode fp32 --eval-set test --i-know-final
python -m src.evaluate --model $M/imp/SPL-b567-k32.onnx --code SPL-b567-k32 --quant-mode static --eval-set test --i-know-final
deactivate
echo "== Hailo $(date '+%F %T')"
export CUDA_VISIBLE_DEVICES=''
source ~/venvs/hailo/bin/activate
H=/mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder
python -m dfc.final_eval --har $H.quantized.har --code H-official --fp-code H-FP --i-know-final
python -m dfc.final_eval --har $H.splf_b7_k32.quantized.har --code H-splf-b7-k32 --i-know-final
echo "== 표 $(date '+%F %T')"
python -m src.coco_retrieval table --ref A0 --codes A0 SPL-b567-k32 --split confirm --margin 2.0 --tag final_ort
python -m src.coco_retrieval table --ref H-FP --codes H-FP H-official H-splf-b7-k32 --split confirm --margin 2.0 --tag final_hailo
ls results/coco_ret_confirm* 2>/dev/null; cat results/coco_ret_confirm*final*.md 2>/dev/null || true
python3 - <<'PY'
import numpy as np, json
from math import comb
def mc(a,b):
    x,y=int((a&~b).sum()),int((~a&b).sum()); n,k=x+y,min(x,y)
    return x,y,(1.0 if n==0 else min(1.0,2*sum(comb(n,i) for i in range(k+1))/2**n))
P="results/preds/"
for ref,c in [("A0","SPL-b567-k32"),("H-FP","H-official"),("H-FP","H-splf-b7-k32")]:
    r=np.load(P+f"{ref}-test.npz"); d=np.load(P+f"{c}-test.npz")
    a=r["preds"]==r["labels"]; b=d["preds"]==d["labels"]
    print(f"CIFAR test | {ref} {a.mean():.2%} | {c} {b.mean():.2%} | 차이 {100*(b.mean()-a.mean()):+.2f}%p | McNemar {mc(a,b)}")
PY
echo "== 끝 $(date '+%F %T')"
