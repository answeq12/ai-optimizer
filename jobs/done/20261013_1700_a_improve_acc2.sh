# 조합 확인 (정확도만): per-channel + QOperator, INT4 블록32 + accuracy_level 4(INT8 계산, 속도용). 약 15분
source ~/venvs/halio/bin/activate
M=$(python -m src.common models-dir); EX=results/sensitivity/S-c67-fc2.nodes.txt
ev() { [ -f "results/preds/$1-val.json" ] || python -m src.evaluate --model "$2" --code "$1" --quant-mode "$3" --meta "${2%.onnx}.json"; }
[ -f "$M/imp/c67_pc_qop.onnx" ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --quant-format qoperator \
   --calib-list splits/calib_coco_64_s0.txt --exclude-nodes-file $EX --out "$M/imp/c67_pc_qop.onnx"
ev S-c67-pc-qop "$M/imp/c67_pc_qop.onnx" static
[ -f "$M/imp/i4_b32_acc4.onnx" ] || python -m src.quantize_int4 --out "$M/imp/i4_b32_acc4.onnx" --block-size 32 --accuracy-level 4
ev I4-b32-acc4 "$M/imp/i4_b32_acc4.onnx" int4
python -m src.make_table --all > /dev/null
grep -E "^\| (S-c67-pc-qop|I4-b32-acc4)" results/all_runs.md
