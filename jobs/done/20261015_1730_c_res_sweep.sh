# 경량화 2축: 입력 해상도(토큰 수) 축소. 학습 없음. 10/15 투입, 새벽 ORT 측정 뒤에 실행됨
# 크기 192·176·160 (패치 토큰 144·121·100, 224 는 196). 위치 임베딩만 bicubic 보간
# 구성마다 FP32 와 A1-c67-pc(동적 INT8 + 채널별 + fc2 6·7 FP32, 현재 ORT 1순위 후보 레시피) 두 가지
# 평가: CIFAR 검증 10,000 / COCO-select 2,405 (confirm·test 는 열지 않음). 정확도만 (10/16 수정: 속도는 밤 측정 작업으로 분리)
set -e
source ~/venvs/halio/bin/activate
M=$(python -m src.common models-dir); R=$M/res; mkdir -p $R
EX=results/sensitivity/S-c67-fc2.nodes.txt
CODES="A0 A1-c67-pc"; BENCH="A0=$M/clip_vitb16_image_fp32.onnx A1-c67-pc=$M/imp/a1_c67_pc.onnx"
for S in 192 176 160; do
  F=$R/clip_vitb16_image_fp32_r$S.onnx; P=$R/r${S}_prep.onnx; Q=$R/r${S}_a1_c67_pc.onnx
  echo "== r$S: export $(date '+%T')"
  [ -f $F ] || python -m src.export_res --size $S
  [ -f $P ] || python -c "from onnxruntime.quantization.shape_inference import quant_pre_process as q; q('$F','$P',skip_optimization=True)"
  [ -f $Q ] || python -m src.quantize_ort --mode dynamic --per-channel --src $P --exclude-nodes-file $EX --out $Q
  export HALIO_IMAGE_SIZE=$S
  [ -f results/preds/R$S-A0-val.json ] || python -m src.evaluate --model $F --code R$S-A0 --quant-mode fp32
  [ -f results/preds/R$S-A1-c67-pc-val.json ] || python -m src.evaluate --model $Q --code R$S-A1-c67-pc --quant-mode dynamic --meta ${Q%.onnx}.json
  [ -f results/coco_ret/R$S-A0-select.json ] || python -m src.coco_retrieval eval --model $F --code R$S-A0
  [ -f results/coco_ret/R$S-A1-c67-pc-select.json ] || python -m src.coco_retrieval eval --model $Q --code R$S-A1-c67-pc
  unset HALIO_IMAGE_SIZE
  CODES="$CODES R$S-A0 R$S-A1-c67-pc"; BENCH="$BENCH R$S-A0=$F R$S-A1-c67-pc=$Q"
done
python -m src.make_table --all > /dev/null
python -m src.coco_retrieval table --ref A0 --codes $CODES --margin 2.0 --tag res
echo "== CIFAR 검증"; grep -E "^\| (A0|A1-c67-pc|R1[0-9]{2}-)" results/all_runs.md || true
cat results/coco_ret_select_res.md
echo "== 정확도 끝 $(date '+%F %T'). 속도는 jobs/*_d_bench_night.sh 에서 후보들과 한 세션에 잰다"
