# 추가 개선 탐색 (정확도만, 검증셋). 약 50~60분
#  1) 이상치 채널 분리: fc2 입력의 극단값 채널 k개만 FP32 작은 행렬곱으로 떼고 나머지는 INT8 (층 전체 FP32 대신)
#  2) 어텐션 활성값 MatMul(QK^T, AV) 24개도 INT8 (S-c67-pc 위에)
#  3) 동적 INT8 + fc2 6·7 FP32 (민감도 발견이 동적에도 통하는지)
source ~/venvs/halio/bin/activate
M=$(python -m src.common models-dir); mkdir -p "$M/imp"
CAL=splits/calib_coco_64_s0.txt; EX=results/sensitivity/S-c67-fc2.nodes.txt
ev() { [ -f "results/preds/$1-val.json" ] || python -m src.evaluate --model "$2" --code "$1" --quant-mode "$3" --meta "${2%.onnx}.json"; }
split() {  # 코드 블록 k
  local code=$1 b=$2 k=$3 f="$M/imp/split_$1.fp32.onnx" q="$M/imp/$1.onnx"
  echo "== $code: fc2 블록 $b, 이상치 채널 $k 개 분리"
  [ -f "$f" ] || python -m src.split_outliers --blocks "$b" --k "$k" --calib-list $CAL --out "$f" || return
  [ -f "$q" ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --src "$f" \
      --exclude-nodes-file "$f.hot_nodes.txt" --calib-list $CAL --out "$q" || return
  ev "$code" "$q" static
}
split SPL-b67-k4   6,7    4
split SPL-b67-k8   6,7    8
split SPL-b567-k8  5,6,7  8
split SPL-b567-k32 5,6,7  32
echo "== 어텐션 MatMul 포함 (S-c67-pc + QK^T·AV INT8)"
[ -f "$M/imp/c67_pc_attn.onnx" ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --include-attn-matmul \
   --calib-list $CAL --exclude-nodes-file $EX --out "$M/imp/c67_pc_attn.onnx"
ev S-c67-pc-attn "$M/imp/c67_pc_attn.onnx" static
echo "== 동적 INT8 + fc2 6·7 FP32"
[ -f "$M/imp/a1_c67.onnx" ] || python -m src.quantize_ort --mode dynamic --exclude-nodes-file $EX --out "$M/imp/a1_c67.onnx"
ev A1-c67 "$M/imp/a1_c67.onnx" dynamic
[ -f "$M/imp/a1_c67_pc.onnx" ] || python -m src.quantize_ort --mode dynamic --per-channel --exclude-nodes-file $EX --out "$M/imp/a1_c67_pc.onnx"
ev A1-c67-pc "$M/imp/a1_c67_pc.onnx" dynamic
python -m src.make_table --all > /dev/null
echo "== 결과 (검증셋)"
grep -E "^\| (A0|A1|S-c67-pc|SPL-|S-c67-pc-attn|A1-c67)" results/all_runs.md
grep -h "split\]" -r /dev/null; for f in "$M"/imp/split_*.json; do python -c "import json;d=json.load(open('$f'));print('$f'.split('/')[-1], [ (l['block'], round(l['absmax_before'],1), round(l['rest_absmax'],1)) for l in d['layers']], d['size_mb'])"; done
