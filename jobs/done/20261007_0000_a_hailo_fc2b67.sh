# Hailo: 블록 6,7 의 fc2 만 a16_w16 (공식 설정 + COCO 64 s0). 컨텍스트를 13개로 유지하며 정확도를 지키는지. 약 50분
export CUDA_VISIBLE_DEVICES=''
bash dfc/setup_and_smoke.sh compile --calib-list splits/calib_coco_64_s0.txt --tag coco64_fc2b67 \
  --fc2-16bit-blocks 6,7 --eval-n 2000 --save-preds
