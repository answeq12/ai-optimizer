# 점수 올릴 여지 실험 (정확도만, 검증셋). 기존 모델·결과는 건드리지 않고 새 이름으로 저장. 약 40~60분
source ~/venvs/halio/bin/activate
THREADS=${THREADS:-4}
M=$(python -m src.common models-dir); mkdir -p "$M/imp"
EX=results/sensitivity/S-c67-fc2.nodes.txt
ev() { [ -f "results/preds/$1-val.json" ] || python -m src.evaluate --model "$2" --code "$1" --quant-mode "$3" --threads "$THREADS" --meta "${2%.onnx}.json"; }
echo "== ① 혼합 정밀도(fc2 6·7 FP32) + per-channel"
[ -f "$M/imp/c67_pc.onnx" ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel \
   --calib-list splits/calib_coco_64_s0.txt --exclude-nodes-file $EX --out "$M/imp/c67_pc.onnx"
ev S-c67-pc "$M/imp/c67_pc.onnx" static
echo "== ② 혼합 정밀도를 QOperator 형식으로 (정확도 유지 확인, 속도는 나중에)"
[ -f "$M/imp/c67_qop.onnx" ] || python -m src.quantize_ort --mode static --weight-matmul-only --quant-format qoperator \
   --calib-list splits/calib_coco_64_s0.txt --exclude-nodes-file $EX --out "$M/imp/c67_qop.onnx"
ev S-c67-qop "$M/imp/c67_qop.onnx" static
echo "== ③ 가중치 전용 INT4 (활성값 FP32)"
[ -f "$M/imp/i4_all.onnx" ]  || python -m src.quantize_int4 --out "$M/imp/i4_all.onnx"
ev I4-all "$M/imp/i4_all.onnx" int4
[ -f "$M/imp/i4_b32.onnx" ]  || python -m src.quantize_int4 --out "$M/imp/i4_b32.onnx" --block-size 32
ev I4-b32 "$M/imp/i4_b32.onnx" int4
[ -f "$M/imp/i4_ex67.onnx" ] || python -m src.quantize_int4 --out "$M/imp/i4_ex67.onnx" --exclude-nodes-file $EX
ev I4-ex67 "$M/imp/i4_ex67.onnx" int4
python -m src.make_table --all > /dev/null
echo "== 결과 (검증셋)"
grep -E "^\| (A0|A1|S-c67-fc2|S-c67-pc|S-c67-qop|I4-)" results/all_runs.md
for f in "$M"/imp/*.json; do python -c "import json,sys;d=json.load(open('$f'));print('$f'.split('/')[-1], d.get('size_mb'), 'MB', d.get('n_matmulnbits_in_output',''))"; done
