# E47 보고용 평가 (동결 후, 선택에 쓰지 않음): DA5-A0·DA5-SPL 의 COCO-confirm·CIFAR test·Flickr30k·NoCaps (이미 열린 셋에 대한 보고값)
set -eo pipefail
source ~/venvs/halio/bin/activate
S=/mnt/d/halio_data; M=$S/models_da5
export HALIO_MODELS_SUBDIR=models_da5
for c in "DA5-A0 $M/clip_vitb16_image_fp32.onnx fp32" "DA5-SPL $M/imp/SPL-b567-k32.onnx static"; do set -- $c
  python -m src.coco_retrieval eval --model $2 --code $1 --split confirm --i-know-final | tail -1
  python -m src.evaluate --model $2 --code $1 --quant-mode $3 --eval-set test --i-know-final | tail -1
done
unset HALIO_MODELS_SUBDIR
python -m src.coco_retrieval table --ref A0 --codes A0 SPL-b567-k32 DA2-SPL DA5-A0 DA5-SPL --split confirm --margin 2.0 --tag final_da5
python -m src.coco_retrieval table --ref DA5-A0 --codes DA5-A0 DA5-SPL --split confirm --margin 2.0 --tag final_da5_quant
python -m src.flickr_eval text --tag da5 --text-model $M/text_state.pt
python -m src.flickr_eval eval DA5-A0=$M/clip_vitb16_image_fp32.onnx:da5 DA5-SPL=$M/imp/SPL-b567-k32.onnx:da5
python -m src.flickr_eval table --ref A0 --codes A0 SPL-b567-k32 DA2-SPL DA5-A0 DA5-SPL
python -m src.nocaps_eval text --tag da5 --text-model $M/text_state.pt
python -m src.nocaps_eval eval --i-know-final DA5-A0=$M/clip_vitb16_image_fp32.onnx:da5 DA5-SPL=$M/imp/SPL-b567-k32.onnx:da5
python -m src.nocaps_eval table --i-know-final --ref A0 --codes A0 SPL-b567-k32 DA2-SPL DA5-A0 DA5-SPL
for c in A0 SPL-b567-k32 DA2-SPL DA5-A0 DA5-SPL; do python -c "import json;print('CIFAR test $c', round(json.load(open('results/preds/$c-test.json'))['top1_acc']*100,2))"; done
