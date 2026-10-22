# E38 해상도 192 + SPL 정적 (약 1시간): 원본·적응 두 모델, select·CIFAR val 만
set -eo pipefail
source ~/venvs/halio/bin/activate
S=/mnt/d/halio_data; CAL=splits/calib_coco_64_s0.txt
[ -f $S/models/res/clip_vitb16_image_fp32_r192.onnx ] || python -m src.export_res --size 192
for V in r192 da_r192; do
  M=$S/models_$V; mkdir -p $M/imp
  cp -n $S/models/text_emb_cifar100.npy $S/models/cifar100_classes.json $S/models/coco_ret_text_select.npy $S/models/coco_ret_text_select.json $M/
  if [ $V = r192 ]; then cp -n $S/models/res/clip_vitb16_image_fp32_r192.onnx $M/clip_vitb16_image_fp32.onnx; A0=R192-A0; Q=R192-SPL
  else [ -f $M/clip_vitb16_image_fp32.onnx ] || python -m src.export_res --size 192 --tail $S/models_da/da_tail_chosen.pt --out $M/clip_vitb16_image_fp32.onnx; A0=R192-DA-A0; Q=R192-DA-SPL; fi
  export HALIO_MODELS_SUBDIR=models_$V HALIO_IMAGE_SIZE=192
  F=$M/imp/split_SPL-b567-k32.fp32.onnx
  [ -f $F ] || python -m src.split_outliers --blocks 5,6,7 --k 32 --calib-list $CAL --out $F
  [ -f $M/imp/SPL-b567-k32.onnx ] || python -m src.quantize_ort --mode static --weight-matmul-only --per-channel --src $F --exclude-nodes-file $F.hot_nodes.txt --calib-list $CAL --out $M/imp/SPL-b567-k32.onnx
  for c in "$A0 $M/clip_vitb16_image_fp32.onnx fp32" "$Q $M/imp/SPL-b567-k32.onnx static"; do set -- $c
    [ -f results/preds/$1-val.json ] || python -m src.evaluate --model $2 --code $1 --quant-mode $3 --eval-set val | tail -1
    [ -f results/coco_ret/$1-select.json ] || python -m src.coco_retrieval eval --model $2 --code $1 --split select | tail -1
  done
  unset HALIO_MODELS_SUBDIR HALIO_IMAGE_SIZE
done
python -m src.coco_retrieval table --ref A0 --codes A0 SPL-b567-k32 R192-A0 R192-SPL DA-A0 DA-SPL R192-DA-A0 R192-DA-SPL --split select --margin 2.0 --tag e38
python -m src.coco_retrieval table --ref DA-A0 --codes DA-A0 DA-SPL R192-DA-A0 R192-DA-SPL --split select --margin 2.0 --tag e38_da
