# 사후 분석 (10/18 투입). 최종 판정(10/17)과 구성 동결 이후의 보고용 평가. 선택·규칙을 바꾸지 않는다
# 목적: 단계별 기여도 표를 COCO-confirm 으로 보고 (A2 → A2m → +fc2 보호 → +per-channel → 이상치 채널 분리)
# 이미 있는 confirm 결과(A0, SPL-b567-k32)는 다시 돌리지 않음. 정확도 작업이라 PC 조용할 필요 없음. 예상 30~60분
set -e
source ~/venvs/halio/bin/activate
M=$(python -m src.common models-dir)
set +e
ev() { [ -f "results/coco_ret/$1-confirm.json" ] || python -m src.coco_retrieval eval --model "$2" --code "$1" --split confirm --i-know-final; }
ev A1            "$M/a1_dynamic.onnx"
ev A2            "$M/a2_static_minmax.onnx"
ev A2m           "$M/a2m_minmax.onnx"
ev S-c67-fc2     "$M/sens/S-c67-fc2.onnx"
ev S-c67-pc      "$M/imp/c67_pc.onnx"
ev A1-c67-pc     "$M/imp/a1_c67_pc.onnx"
set -e
python -m src.coco_retrieval table --ref A0 --codes A0 A1 A2 A2m S-c67-fc2 S-c67-pc SPL-b567-k32 A1-c67-pc --split confirm --margin 2.0 --tag posthoc_ablation_ort
cat results/coco_ret_confirm*posthoc_ablation_ort*.md 2>/dev/null || true
echo "== 끝 $(date '+%F %T')"
