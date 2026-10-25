# 재현 패키지: 최종 적응 가중치(DA5) 넣기 + sha256. 이전(DA, E33) 가중치는 _archive 로
set -eo pipefail
mkdir -p _archive/repro_da1/weights
[ -f repro/weights/da_tail_chosen.pt ] && mv -n repro/weights/da_tail_chosen.pt repro/weights/da_tail_chosen.json repro/weights/SHA256SUMS _archive/repro_da1/weights/ || true
cp /mnt/d/halio_data/models_da/da2_chosen_e46.pt repro/weights/da5_chosen.pt
echo '{"experiment": "E46", "data": "COCO train2017 118,287 + CC3M 100,000", "alpha": 1.0, "ckpt": "da2_e46_ep2.pt", "keys": ["img", "txt"]}' > repro/weights/da5_chosen.json
(cd repro/weights && sha256sum da5_chosen.pt > SHA256SUMS && cat SHA256SUMS && ls -la)
