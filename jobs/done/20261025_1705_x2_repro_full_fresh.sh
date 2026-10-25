# 재현 검증: 빈 저장 폴더에서 reproduce.sh 전체 (정확도만, 속도 생략). 결과 코드 RV-*
set -eo pipefail
export HALIO_DATA_DIR=/mnt/d/halio_repro_check_20261025; mkdir -p $HALIO_DATA_DIR
SKIP_BENCH=1 PFX=RV bash reproduce.sh
