# 작업 3(일부): Hailo 에뮬레이터 COCO-select 검색 평가. 공식(FP 포함) / CIFAR 캘리브 / fc2 6·7 a16_w16. 약 40~50분
source ~/venvs/hailo/bin/activate
export CUDA_VISIBLE_DEVICES=''
python -m dfc.coco_ret_eval --har /mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder.quantized.har --code H-official --fp-code H-FP
python -m dfc.coco_ret_eval --har /mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder.cifar64_s0.quantized.har --code H-cifar
python -m dfc.coco_ret_eval --har /mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder.coco64_fc2b67.quantized.har --code H-fc2b67
python -m src.coco_retrieval table --ref H-FP --codes H-FP H-official H-cifar H-fc2b67 --tag hailo
