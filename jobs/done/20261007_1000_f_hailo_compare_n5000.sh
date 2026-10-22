source ~/venvs/hailo/bin/activate
python -m dfc.compare_preds logs/dfc/preds.coco64_base_n5000.npz logs/dfc/preds.coco64_fc2b67_n5000.npz
cp logs/dfc/compare_preds.md logs/dfc/compare_preds_n5000.md
