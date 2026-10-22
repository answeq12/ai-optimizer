# 캘리브레이션 A/B 를 A2m 기준으로 다시 (예전 AB-* 는 무너진 A2 위라 무효). 정확도만. 약 45분
#  - 도메인: COCO s1,s2 (s0 = A2m 자체) / CIFAR s0,s1,s2
#  - 범위 추정: entropy (A2m 과 같은 대상·캘리브레이션)
#  - 혼합 정밀도 후보(fc2 b6,7 FP32)에 CIFAR s0: Hailo 에서처럼 효과가 더해지지 않는지
set -e
source ~/venvs/halio/bin/activate
THREADS=${THREADS:-4}
M=$(python -m src.common models-dir); mkdir -p "$M/ab2"
one() {  # 코드 출력파일 [양자화 인자...]
  local code=$1 out="$M/ab2/$2"; shift 2
  [ -f "$out" ] || python -m src.quantize_ort --mode static --weight-matmul-only --out "$out" "$@"
  [ -f "results/preds/$code-val.json" ] || python -m src.evaluate --model "$out" --code "$code" \
      --quant-mode static --threads "$THREADS" --meta "${out%.onnx}.json"
}
for s in 1 2; do one AB2-coco-s$s coco_s$s.onnx --calib-list splits/calib_coco_64_s$s.txt; done
for s in 0 1 2; do one AB2-cifar-s$s cifar_s$s.onnx --calib-list splits/calib_cifar_64_s$s.txt; done
one A2m-ent a2m_entropy.onnx --calib-list splits/calib_coco_64_s0.txt --calib-method entropy
one S-c67-cifar c67_cifar_s0.onnx --calib-list splits/calib_cifar_64_s0.txt --exclude-nodes-file results/sensitivity/S-c67-fc2.nodes.txt
python -m src.make_table --all > /dev/null
echo "== 결과"
grep -E "^\| (A2m|AB2-|A2m-ent|S-c67)" results/all_runs.md
