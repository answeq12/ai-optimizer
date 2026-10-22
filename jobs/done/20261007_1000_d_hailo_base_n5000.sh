# Hailo 기준선(공식 설정)을 검증셋 5000장으로 평가 (2000장보다 정밀, 보고서용). 약 40분
export CUDA_VISIBLE_DEVICES=''
bash dfc/setup_and_smoke.sh smoke --from-har /mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder.quantized.har --tag coco64_base --eval-n 5000 --save-preds
