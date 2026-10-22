# 견고성 확인: 혼합 정밀도 후보(fc2 b6,7 FP32)를 COCO 다른 묶음 s1, s2 로 캘리브레이션해도 같은 결과인지. 약 12분
set -e
source ~/venvs/halio/bin/activate
M=$(python -m src.common models-dir); mkdir -p "$M/ab2"
for s in 1 2; do
  code=S-c67-coco-s$s; out="$M/ab2/c67_coco_s$s.onnx"
  [ -f "$out" ] || python -m src.quantize_ort --mode static --weight-matmul-only --calib-list splits/calib_coco_64_s$s.txt \
      --exclude-nodes-file results/sensitivity/S-c67-fc2.nodes.txt --out "$out"
  [ -f "results/preds/$code-val.json" ] || python -m src.evaluate --model "$out" --code "$code" --quant-mode static --meta "${out%.onnx}.json"
done
python -m src.make_table --all > /dev/null
grep -E "^\| (S-c67)" results/all_runs.md
