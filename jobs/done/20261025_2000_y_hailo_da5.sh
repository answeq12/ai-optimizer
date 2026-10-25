# 본선 준비 E48: DA5 적응 가중치를 Hailo 로 (공식 ONNX 에 덮어쓰기 → 같은 처방 splf_b7_k32 → optimize·compile → select 에뮬레이터 평가). DFC finetune 안 씀(공식 alls 그대로)
set -eo pipefail
source ~/venvs/halio/bin/activate
python -m dfc.inject_da --models-subdir models_da5 --tag da5
deactivate
export CUDA_VISIBLE_DEVICES=''
source ~/venvs/hailo/bin/activate
S=/mnt/d/halio_data; H=$S/hailo/clip_vit_b_16_image_encoder
timeout -k 5m 4h python -m dfc.split_hailo --blocks 7 --k 32 --tag splf_b7_k32_da5 --split-fc1 --src-onnx $S/hailo/official_da5.onnx
HALIO_MODELS_SUBDIR=models_da5 timeout -k 5m 3h python -m dfc.coco_ret_eval --har $H.splf_b7_k32_da5.quantized.har --code H-DA5-splf --fp-code H-DA5-FP
deactivate; source ~/venvs/halio/bin/activate
python -m src.coco_retrieval table --ref DA5-A0 --codes DA5-A0 DA5-SPL H-DA5-FP H-DA5-splf --split select --margin 2.0 --tag hailo_da5
python -m src.coco_retrieval table --ref H-FP --codes H-FP H-official H-splf-b7-k32 --split select --margin 2.0 --tag hailo_orig_ref
