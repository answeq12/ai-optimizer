# E42 최종 시험 (약 45분): 동결 구성 DA2-SPL 과 짝 FP32, 기준 A0·SPL·DA-SPL. confirm·CIFAR test·Flickr30k·NoCaps 를 한 번에 연다 (결과와 무관하게 보고)
set -eo pipefail
source ~/venvs/halio/bin/activate
S=/mnt/d/halio_data; O=$S/models; D=$S/models_da; M=$S/models_da2
export HALIO_MODELS_SUBDIR=models_da2
for c in "DA2-A0 $M/clip_vitb16_image_fp32.onnx fp32" "DA2-SPL $M/imp/SPL-b567-k32.onnx static"; do set -- $c
  python -m src.coco_retrieval eval --model $2 --code $1 --split confirm --i-know-final | tail -1
  python -m src.evaluate --model $2 --code $1 --quant-mode $3 --eval-set test --i-know-final | tail -1
done
unset HALIO_MODELS_SUBDIR
python -m src.coco_retrieval table --ref A0 --codes A0 SPL-b567-k32 DA-SPL DA2-A0 DA2-SPL --split confirm --margin 2.0 --tag final
python -m src.coco_retrieval table --ref DA2-A0 --codes DA2-A0 DA2-SPL --split confirm --margin 2.0 --tag final_quant
echo "== Flickr30k 1K"
python -m src.flickr_eval text --tag da2 --text-model $M/text_state.pt
python -m src.flickr_eval eval DA2-A0=$M/clip_vitb16_image_fp32.onnx:da2 DA2-SPL=$M/imp/SPL-b567-k32.onnx:da2
python -m src.flickr_eval table --ref A0 --codes A0 SPL-b567-k32 DA-A0 DA-SPL DA2-A0 DA2-SPL
echo "== NoCaps val (최종 테스트셋, 처음이자 마지막으로 연다)"
python -m src.nocaps_eval text --tag da2 --text-model $M/text_state.pt
python -m src.nocaps_eval eval --i-know-final A0=$O/clip_vitb16_image_fp32.onnx SPL-b567-k32=$O/imp/SPL-b567-k32.onnx DA-SPL=$D/imp/SPL-b567-k32.onnx DA2-A0=$M/clip_vitb16_image_fp32.onnx:da2 DA2-SPL=$M/imp/SPL-b567-k32.onnx:da2
python -m src.nocaps_eval table --i-know-final --ref A0 --codes A0 SPL-b567-k32 DA-SPL DA2-A0 DA2-SPL
for c in A0 SPL-b567-k32 DA-SPL DA2-A0 DA2-SPL; do python -c "import json;print('CIFAR test $c', round(json.load(open('results/preds/$c-test.json'))['top1_acc']*100,2))"; done
