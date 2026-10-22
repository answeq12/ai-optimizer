# ORT 민감도 결과의 Hailo 이식: 블록 5,6,7 의 fc2 를 a16_w16 으로 남기고 (그 외 공식 설정 그대로, COCO 64 s0)
# optimize -> 에뮬레이터 2000장(예측 저장) -> Hailo-10H 컴파일. 약 50~60분
export CUDA_VISIBLE_DEVICES=''
bash dfc/setup_and_smoke.sh compile --calib-list splits/calib_coco_64_s0.txt --tag coco64_fc2b567 \
  --fc2-16bit-blocks 5,6,7 --eval-n 2000 --save-preds
