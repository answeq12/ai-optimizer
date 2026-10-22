# E41 Hailo 캘리브 train2017 256장 (약 3시간, 정확도만·select 만)
export CUDA_VISIBLE_DEVICES=''
export HALIO_CALIBSET_SIZE=256
H=/mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder
L=splits/calib_cocotrain_256.txt
bash dfc/setup_and_smoke.sh smoke --calib-list $L --tag cocot256_base --eval-n 300 || echo "✘ base"
bash dfc/setup_and_smoke.sh smoke --calib-list $L --tag cocot256_fc2b7 --fc2-16bit-blocks 7 --eval-n 300 || echo "✘ fc2b7"
source ~/venvs/hailo/bin/activate
python -m dfc.split_hailo --blocks 7 --k 32 --tag splf_b7_k32_t256 --split-fc1 --skip-compile --calib-list $L || echo "✘ splf"
for p in "cocot256_base H-official-t256" "cocot256_fc2b7 H-fc2b7-t256" "splf_b7_k32_t256 H-splf-b7-k32-t256"; do
  set -- $p
  [ -f $H.$1.quantized.har ] || { echo "✘ $1 HAR 없음"; continue; }
  python -m dfc.coco_ret_eval --har $H.$1.quantized.har --code $2 || true
done
C="H-FP H-official H-official-c189 H-official-t256 H-fc2b7 H-fc2b7-c189 H-fc2b7-t256 H-splf-b7-k32 H-splf-b7-k32-c189 H-splf-b7-k32-t256"
have=""; for c in $C; do [ -f results/coco_ret/$c-select.npz ] && have="$have $c"; done
python -m src.coco_retrieval table --ref H-FP --codes $have --split select --margin 2.0 --tag hailo_t256
grep -h "^| H" results/coco_ret_select_hailo_t256.md
