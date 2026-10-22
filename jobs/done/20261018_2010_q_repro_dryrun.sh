# 재현 스크립트 점검 (10/18). 새 PC 를 흉내 내려고 빈 데이터 폴더(/mnt/d/halio_repro_test)에서 reproduce.sh 를 처음부터 돌린다
# 다운로드 약 2GB, 2~3시간. 속도 측정은 생략(SKIP_BENCH=1). 분할 파일은 재생성 후 sha256 비교, 다르면 원본 복원
# 끝나면 /mnt/d/halio_repro_test 는 지워도 됨 (원래 데이터 /mnt/d/halio_data 는 건드리지 않음)
set -e
mkdir -p /mnt/d/halio_repro_test
HALIO_DATA_DIR=/mnt/d/halio_repro_test SKIP_BENCH=1 bash reproduce.sh
