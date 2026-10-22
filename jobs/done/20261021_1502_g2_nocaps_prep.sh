# E34 NoCaps 준비 재시도: venv 생성 불가(python3-venv 없음) → pyarrow 만 별도 폴더(~/pylibs_data)에 설치해 PYTHONPATH 로 사용. 메인 환경은 그대로
set -eo pipefail
source ~/venvs/halio/bin/activate
[ -d ~/pylibs_data/pyarrow ] || pip install -q --target ~/pylibs_data pyarrow
PYTHONPATH=~/pylibs_data python src/nocaps_prep.py /mnt/d/halio_data/data/nocaps
python -m src.nocaps_eval text
du -sh /mnt/d/halio_data/data/nocaps
