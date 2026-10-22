# E26 재실행 (10/18 18:05): 첫 실행은 공식 alls 의 calibset_size=64 때문에 189장 중 앞 64장(=s0)만 쓰여 무효. HALIO_CALIBSET_SIZE=189 로 다시. 정확도만, 약 2.5시간
export CUDA_VISIBLE_DEVICES=''
export HALIO_CALIBSET_SIZE=189
H=/mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder
L=splits/calib_coco_192_s012.txt
bash dfc/setup_and_smoke.sh smoke --calib-list $L --tag coco189_base --eval-n 300 || echo "✘ base192"
bash dfc/setup_and_smoke.sh smoke --calib-list $L --tag coco189_fc2b7 --fc2-16bit-blocks 7 --eval-n 300 || echo "✘ fc2b7 192"
source ~/venvs/hailo/bin/activate
python -m dfc.split_hailo --blocks 7 --k 32 --tag splf_b7_k32_c189 --split-fc1 --skip-compile --calib-list $L || echo "✘ splf 192"
for p in "coco189_base H-official-c189" "coco189_fc2b7 H-fc2b7-c189" "splf_b7_k32_c189 H-splf-b7-k32-c189"; do
  set -- $p
  [ -f $H.$1.quantized.har ] || { echo "✘ $1 HAR 없음"; continue; }
  python -m dfc.coco_ret_eval --har $H.$1.quantized.har --code $2 || true
  python -m dfc.final_eval --har $H.$1.quantized.har --code $2 --skip-cifar --i-know-final || true
done
C="H-FP H-official H-official-c189 H-fc2b7 H-fc2b7-s1 H-fc2b7-s2 H-fc2b7-c189 H-splf-b7-k32 H-splf-b7-k32-s1 H-splf-b7-k32-s2 H-splf-b7-k32-c189"
for sp in select confirm; do
  have=""; for c in $C; do [ -f results/coco_ret/$c-$sp.npz ] && have="$have $c"; done
  python -m src.coco_retrieval table --ref H-FP --codes $have --split $sp --margin 2.0 --tag calib189
done
grep -h "^| H" results/coco_ret_*calib189*.md
