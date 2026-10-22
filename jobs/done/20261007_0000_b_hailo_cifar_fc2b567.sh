# Hailo: CIFAR 64 s0 캘리브레이션 + 블록 5,6,7 fc2 a16_w16. 두 개선이 더해지는지. 약 50분
export CUDA_VISIBLE_DEVICES=''
bash dfc/setup_and_smoke.sh compile --calib-list splits/calib_cifar_64_s0.txt --tag cifar64_fc2b567 \
  --fc2-16bit-blocks 5,6,7 --eval-n 2000 --save-preds
