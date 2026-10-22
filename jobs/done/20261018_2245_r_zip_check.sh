# 재현 zip 점검 (10/18 22:45). dist/halio_repro_20261018.zip 을 새 폴더(~/halio_zipcheck)에 풀어 그 안에서만 실행한다
#  1) SELFTEST (가짜 데이터, 2~3분)  2) reproduce.sh 전체 (SKIP_BENCH=1)
#  데이터·캐시는 job q 가 받아 둔 /mnt/d/halio_repro_test 를 링크로 재사용(다운로드 없음), 모델 폴더는 새로 → 양자화부터 zip 코드로 다시 만든다
#  원래 프로젝트 폴더·/mnt/d/halio_data 는 건드리지 않는다. 약 1.5시간. 끝나면 ~/halio_zipcheck, /mnt/d/halio_zipcheck 는 지워도 됨
set -e
Z="$PWD/dist/halio_repro_20261018.zip"; W=~/halio_zipcheck; S=/mnt/d/halio_zipcheck
[ -f "$Z" ] || { echo "✘ zip 없음: $Z"; exit 1; }
[ -e "$W" ] && mv "$W" "${W}_old_$(date +%H%M%S)"
mkdir -p "$W" "$S"
python3 -m zipfile -e "$Z" "$W"
cd "$W/halio_project"; echo "== 풀린 파일 $(find . -type f | wc -l)개, 위치 $PWD"
for d in data cache; do [ -e "$S/$d" ] || ln -s /mnt/d/halio_repro_test/$d "$S/$d"; done
ls -la "$S"
echo "== 1) SELFTEST"
SELFTEST=1 bash run_pilot.sh > selftest.log 2>&1 && echo "   SELFTEST 통과" || { echo "✘ SELFTEST 실패"; tail -30 selftest.log; }
echo "== 2) reproduce.sh (zip 코드, 새 모델 폴더)"
HALIO_DATA_DIR="$S" SKIP_BENCH=1 bash reproduce.sh
