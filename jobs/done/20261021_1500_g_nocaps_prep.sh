# E34 최종 테스트셋 NoCaps val 준비 (약 20~30분): 별도 venv(~/venvs/data)로 받기 → 원본 텍스트 인코더로 캡션 임베딩. 이미지 평가는 하지 않음(봉인)
set -eo pipefail
[ -x ~/venvs/data/bin/python ] || { python3 -m venv ~/venvs/data; ~/venvs/data/bin/pip install -q pyarrow huggingface_hub; }
~/venvs/data/bin/python src/nocaps_prep.py /mnt/d/halio_data/data/nocaps
source ~/venvs/halio/bin/activate
python -m src.nocaps_eval text
ls -la /mnt/d/halio_data/data/nocaps | head; du -sh /mnt/d/halio_data/data/nocaps
