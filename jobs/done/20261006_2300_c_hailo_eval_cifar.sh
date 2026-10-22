# CIFAR 64 캘리브레이션 양자화 HAR 을 같은 2000장으로 평가. 약 25분
export CUDA_VISIBLE_DEVICES=''
bash dfc/setup_and_smoke.sh smoke --from-har /mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder.cifar64_s0.quantized.har --tag cifar64_base --eval-n 2000 --save-preds
