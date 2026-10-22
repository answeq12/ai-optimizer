# E24 사후 분석 (10/17 22:35, 동결 후 원인 확인용 — 최종 후보는 바꾸지 않는다)
# 질문: H-splf-b7-k32 가 select −1.10 → confirm −3.03 으로 떨어진 이유가 선택 편향인가, 캘리브 묶음 변동인가?
#  1) H-fc2b7(층 전체 16비트, s0) confirm 1회
#  2) 캘리브 s1·s2 로 H-fc2b7 / H-splf-b7-k32 를 다시 양자화(컴파일 없음) → select + confirm
# 정확도만. 약 3시간
export CUDA_VISIBLE_DEVICES=''
source ~/venvs/hailo/bin/activate
H=/mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder
echo "== 1) H-fc2b7 confirm $(date '+%T')"
python -m dfc.final_eval --har $H.coco64_fc2b7.quantized.har --code H-fc2b7 --skip-cifar --i-know-final || echo "✘ fc2b7 confirm"
for S in s1 s2; do
  echo "== 2) seed $S $(date '+%T')"
  deactivate; bash dfc/setup_and_smoke.sh smoke --calib-list splits/calib_coco_64_$S.txt --tag coco64_${S}_fc2b7 --fc2-16bit-blocks 7 --eval-n 300 || echo "✘ fc2b7 $S"
  source ~/venvs/hailo/bin/activate
  if [ -f $H.coco64_${S}_fc2b7.quantized.har ]; then
    python -m dfc.coco_ret_eval --har $H.coco64_${S}_fc2b7.quantized.har --code H-fc2b7-$S || true
    python -m dfc.final_eval --har $H.coco64_${S}_fc2b7.quantized.har --code H-fc2b7-$S --skip-cifar --i-know-final || true
  fi
  python -m dfc.split_hailo --blocks 7 --k 32 --tag splf_b7_k32_$S --split-fc1 --skip-compile --calib-list splits/calib_coco_64_$S.txt || echo "✘ splf $S"
  if [ -f $H.splf_b7_k32_$S.quantized.har ]; then
    python -m dfc.coco_ret_eval --har $H.splf_b7_k32_$S.quantized.har --code H-splf-b7-k32-$S || true
    python -m dfc.final_eval --har $H.splf_b7_k32_$S.quantized.har --code H-splf-b7-k32-$S --skip-cifar --i-know-final || true
  fi
done
echo "== 표 $(date '+%T')"
C="H-FP H-official H-fc2b7 H-fc2b7-s1 H-fc2b7-s2 H-splf-b7-k32 H-splf-b7-k32-s1 H-splf-b7-k32-s2"
for sp in select confirm; do
  have=""; for c in $C; do [ -f results/coco_ret/$c-$sp.npz ] && have="$have $c"; done
  python -m src.coco_retrieval table --ref H-FP --codes $have --split $sp --margin 2.0 --tag posthoc_seeds
done
cat results/coco_ret_*posthoc_seeds*.md 2>/dev/null || true
echo "== 끝 $(date '+%F %T')"
