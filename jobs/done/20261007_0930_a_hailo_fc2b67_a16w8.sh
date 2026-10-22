# Hailo: 블록 6,7 fc2 를 a16_w8 (활성값 16비트, 가중치 8비트). 컨텍스트가 13개로 돌아오면서 정확도를 지키는지. 약 50분
export CUDA_VISIBLE_DEVICES=''
bash dfc/setup_and_smoke.sh compile --calib-list splits/calib_coco_64_s0.txt --tag coco64_fc2b67_a16w8 \
  --fc2-16bit-blocks 6,7 --fc2-precision a16_w8 --eval-n 2000 --save-preds
source ~/venvs/hailo/bin/activate
python -m dfc.compare_preds logs/dfc/preds.coco64_base_n2000.npz logs/dfc/preds.coco64_fc2b67_n2000.npz \
  logs/dfc/preds.coco64_fc2b67_a16w8_n2000.npz
echo "== 컨텍스트 수: $(grep -c 'Compiling kernels of' "$0" 2>/dev/null)"
