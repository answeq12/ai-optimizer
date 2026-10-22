# E23 변형: 논문 설정(top_layer 5)과 뉴런 수를 바꿔 테스트 시 레지스터의 정확도 유지 여부 확인. 정확도만, 각 약 5분
source ~/venvs/halio/bin/activate
python -m src.register_probe --top-layer 5 --top-k 10 --tag _L5k10
python -m src.register_probe --top-layer 5 --top-k 4 --tag _L5k4
python -m src.register_probe --top-layer 6 --top-k 6 --tag _L6k6
python -m src.register_probe --top-layer 7 --top-k 4 --tag _L7k4
grep -h -E "^# |테스트 시 레지스터|^\| (5|6|7) \|" results/register_probe_L*.md
