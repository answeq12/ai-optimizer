# 작업 1: COCO 텍스트→이미지 검색 평가 (ORT, COCO-select). 약 40~50분
set -e
source ~/venvs/halio/bin/activate
M=$(python -m src.common models-dir)
python -m src.coco_retrieval prepare
python -m src.coco_retrieval text
set +e
ev() { [ -f "results/coco_ret/$1-select.json" ] || python -m src.coco_retrieval eval --model "$2" --code "$1"; }
ev A0            "$M/clip_vitb16_image_fp32.onnx"
ev A1            "$M/a1_dynamic.onnx"
ev A2            "$M/a2_static_minmax.onnx"
ev A2m           "$M/a2m_minmax.onnx"
ev S-c67-fc2     "$M/sens/S-c67-fc2.onnx"
ev S-c67-pc      "$M/imp/c67_pc.onnx"
ev SPL-b567-k32  "$M/imp/SPL-b567-k32.onnx"
ev S-c67-pc-attn "$M/imp/c67_pc_attn.onnx"
ev A1-c67-pc     "$M/imp/a1_c67_pc.onnx"
python -m src.coco_retrieval table --ref A0 --codes A0 A1 A2 A2m S-c67-fc2 S-c67-pc SPL-b567-k32 S-c67-pc-attn A1-c67-pc
