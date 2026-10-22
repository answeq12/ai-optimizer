#!/usr/bin/env bash
# 작업 대기소. jobs/ 에 .sh 가 들어오면 이름 순서대로 하나씩 실행한다.
# 이 창을 열어두면 이후 실험은 더블클릭 없이 돌아간다. 한 번에 하나만 돌기 때문에
# ORT 실험과 DFC 컴파일이 동시에 돌아 메모리를 나눠 쓰는 일도 생기지 않는다.
#   jobs/<이름>.sh        대기 중
#   jobs/running.txt      지금 도는 작업
#   jobs/done/<이름>.sh   끝난 작업 (지우지 않는다)
#   jobs/logs/<이름>.log  작업 출력 + 마지막 줄 rc=<종료코드>
#   jobs/STOP             이 파일이 생기면 지금 작업을 마친 뒤 대기소 종료
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
J="$HERE/jobs"
mkdir -p "$J/done" "$J/logs"
rm -f "$J/STOP" "$J/running.txt"
echo "==================================================================="
echo " halio 작업 대기소 시작  $(date '+%F %T')"
echo " 이 창을 닫지 마세요. 최소화는 괜찮습니다."
echo " 감시 폴더: $J"
echo "==================================================================="
while true; do
  date '+%s' > "$J/.heartbeat"
  [ -e "$J/STOP" ] && { echo "[$(date '+%H:%M:%S')] STOP 파일 확인. 대기소 종료"; rm -f "$J/STOP"; exit 0; }
  f=$(ls "$J"/*.sh 2>/dev/null | sort | head -1)
  if [ -n "$f" ]; then
    n=$(basename "$f" .sh)
    mv "$f" "$J/done/$n.sh"
    echo "$n $(date '+%F %T')" > "$J/running.txt"
    echo "[$(date '+%H:%M:%S')] >>> 실행: $n"
    ( cd "$HERE" && bash "$J/done/$n.sh" ) < /dev/null 2>&1 | tee "$J/logs/$n.log"
    rc=${PIPESTATUS[0]}
    echo "rc=$rc" >> "$J/logs/$n.log"
    rm -f "$J/running.txt"
    echo "[$(date '+%H:%M:%S')] <<< 완료: $n (종료코드 $rc)"
    continue
  fi
  sleep 15
done
