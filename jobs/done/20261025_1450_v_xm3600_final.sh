# E43/E47 최종 시험: XM3600 (사용자 확인 10/25) — 한 번만
set -eo pipefail
source ~/venvs/halio/bin/activate
D=/mnt/d/halio_data
python -m src.xm3600_eval text
python -m src.xm3600_eval text --tag da2 --text-model $D/models_da2/text_state.pt
python -m src.xm3600_eval text --tag da3 --text-model $D/models_da3/text_state.pt
python -m src.xm3600_eval text --tag da5 --text-model $D/models_da5/text_state.pt
python -m src.xm3600_eval eval --i-know-final A0=$D/models/clip_vitb16_image_fp32.onnx SPL=$D/models/imp/SPL-b567-k32.onnx \
  DA2-SPL=$D/models_da2/imp/SPL-b567-k32.onnx:da2 DA3-SPL=$D/models_da3/imp/SPL-b567-k32.onnx:da3 \
  DA5-A0=$D/models_da5/clip_vitb16_image_fp32.onnx:da5 DA5-SPL=$D/models_da5/imp/SPL-b567-k32.onnx:da5
python -m src.xm3600_eval table --i-know-final --ref A0 --codes A0 SPL DA2-SPL DA3-SPL DA5-A0 DA5-SPL --pair DA5-SPL,DA2-SPL
