# XM3600 살펴보기 재실행 (사진이 하위 폴더 없이 풀려 경로 찾기 수정). 모델 평가 없음
set -eo pipefail
source ~/venvs/halio/bin/activate
python -m src.xm3600_prep inspect
