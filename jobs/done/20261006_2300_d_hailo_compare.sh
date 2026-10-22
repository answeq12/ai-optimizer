# 세 결과를 같은 2000장으로 짝지어 비교 -> logs/dfc/compare_preds.md
source ~/venvs/hailo/bin/activate
python -m dfc.compare_preds logs/dfc/preds.coco64_base_n2000.npz logs/dfc/preds.cifar64_base_n2000.npz logs/dfc/preds.coco64_fc2b567_n2000.npz
