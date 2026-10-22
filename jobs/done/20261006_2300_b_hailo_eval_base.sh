# 기준선: 공식 설정(COCO 64) 양자화 HAR 을 같은 2000장으로 다시 평가 (짝지은 비교용). 약 25분
export CUDA_VISIBLE_DEVICES=''
bash dfc/setup_and_smoke.sh smoke --from-har /mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder.quantized.har --tag coco64_base --eval-n 2000 --save-preds
