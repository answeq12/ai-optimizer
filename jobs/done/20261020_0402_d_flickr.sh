# COCO 밖 확인 (약 20~30분): Flickr30k 1K test 텍스트→이미지. A0·SPL 은 항상, DA 는 모델이 있을 때만
set -eo pipefail
source ~/venvs/halio/bin/activate
python -m src.flickr_eval prepare
python -m src.flickr_eval text
O=/mnt/d/halio_data/models; D=/mnt/d/halio_data/models_da
python -m src.flickr_eval eval A0=$O/clip_vitb16_image_fp32.onnx SPL-b567-k32=$O/imp/SPL-b567-k32.onnx
C="A0 SPL-b567-k32"
if [ -f $D/imp/SPL-b567-k32.onnx ]; then python -m src.flickr_eval eval DA-A0=$D/clip_vitb16_image_fp32.onnx DA-SPL=$D/imp/SPL-b567-k32.onnx; C="$C DA-A0 DA-SPL"; fi
python -m src.flickr_eval table --ref A0 --codes $C
