# job r 에서 SELFTEST 가 'python: command not found' 로 실패(새 터미널에서 파이썬 환경을 안 켠 상태와 같음) → run_pilot.sh 에 환경 자동 켜기 추가 후 zip 재생성.
# 새 zip 을 다른 새 폴더에 풀어 SELFTEST 만 다시 (약 3분). 대기소 셸은 venv 가 꺼진 상태라 새 PC 의 새 터미널과 같은 조건
set -e
Z="$PWD/dist/halio_repro_20261018.zip"; W=~/halio_zipcheck2
[ -e "$W" ] && mv "$W" "${W}_old_$(date +%H%M%S)"
mkdir -p "$W"; python3 -m zipfile -e "$Z" "$W"; cd "$W/halio_project"
command -v python || echo "   (python 없음: 새 터미널과 같은 조건)"
SELFTEST=1 bash run_pilot.sh > selftest.log 2>&1 && echo "   SELFTEST 통과" || echo "✘ SELFTEST 실패"
tail -25 selftest.log
