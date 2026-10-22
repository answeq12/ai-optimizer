# E35 학습 방식 개선 (약 3.5~4시간): a) bank 16384 + multi-positive 2에폭 b) 기존 손실 4에폭. 각자 α 탐색. 꼬리표 이름으로 저장
set -eo pipefail
source ~/venvs/halio/bin/activate
export HALIO_DA_FEAT=$(cat logs/da_feat_dir.txt)
mkdir -p /mnt/d/halio_data/models_da/e33_keep; cp -n /mnt/d/halio_data/models_da/da_tail_*.pt /mnt/d/halio_data/models_da/da_tail_chosen.json /mnt/d/halio_data/models_da/e33_keep/
timeout -k 60 3h python -m src.da_train --n 30000 --tag e35a --bank 16384 --multipos
python -m src.da_sweep --ckpt da_tail_e35a_ep2.pt --tag e35a || echo "(e35a 판단 ① 미통과)"
timeout -k 60 4h python -m src.da_train --n 30000 --tag e35b --epochs 4
python -m src.da_sweep --ckpt da_tail_e35b_ep4.pt --tag e35b || echo "(e35b 판단 ① 미통과)"
grep -h "선택\|판단" results/da_sweep_e35a.md results/da_sweep_e35b.md
