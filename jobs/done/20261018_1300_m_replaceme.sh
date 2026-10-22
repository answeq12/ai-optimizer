# E25 블록 제거 + 선형 대체 (ReplaceMe). 학습 없음. n=1,2,3 (위치 자동) → FP32 COCO-select + CIFAR 검증 10k. 정확도만, 약 1시간
source ~/venvs/halio/bin/activate
M=$(python -m src.common models-dir)
for n in 1 2 3; do python -m src.replaceme --n $n || echo "✘ replaceme n=$n"; done
CODES="A0"
for f in $M/rm/clip_vitb16_rm_n*_s*.onnx; do
  t=$(basename $f .onnx); t=${t#clip_vitb16_rm_}; code=RM-${t//_/-}
  python -m src.coco_retrieval eval --model $f --code $code && CODES="$CODES $code"
  python -m src.evaluate --model $f --code $code --quant-mode fp32 || true
done
python -m src.coco_retrieval table --ref A0 --codes $CODES --margin 2.0 --tag replaceme
cat results/coco_ret_select_replaceme.md
grep -E "^RM-" results/accuracy.csv | cut -d, -f1,16 || true
