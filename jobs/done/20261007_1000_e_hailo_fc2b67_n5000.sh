# Hailo 본선 후보(fc2 b6,7 a16_w16)를 같은 5000장으로 평가. 약 40분
export CUDA_VISIBLE_DEVICES=''
bash dfc/setup_and_smoke.sh smoke --from-har /mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder.coco64_fc2b67.quantized.har --tag coco64_fc2b67 --eval-n 5000 --save-preds
