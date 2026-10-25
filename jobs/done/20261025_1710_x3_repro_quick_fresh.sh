# 재현 검증: 빈 저장 폴더에서 repro_quick.sh (EXPECTED_QUICK 의 DA5 기대값도 이 실행으로 만든다 — 아래 x4)
set -eo pipefail
export HALIO_DATA_DIR=/mnt/d/halio_repro_quick_20261025; mkdir -p $HALIO_DATA_DIR
bash repro_quick.sh || true
