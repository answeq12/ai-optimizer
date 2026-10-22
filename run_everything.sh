#!/usr/bin/env bash
# 파일럿 -> 탐침 -> DFC 스모크 테스트를 차례로 한 번에 돌린다. 자리를 비울 때 쓴다.
#
#   bash run_everything.sh
#
# - 이미 끝난 단계는 건너뛴다 (첫 비교표가 있으면 파일럿을, 전체 표가 있으면 탐침을 건너뛴다)
# - 파일럿이 실패하면 거기서 멈춘다. 탐침이나 DFC 가 실패해도 그 앞 단계의 결과는 남는다
# - DFC 단계는 시스템 패키지 설치 때문에 WSL 비밀번호를 물을 수 있다. 그때는 입력할 때까지 기다린다
# - 속도·메모리를 재는 동안에는 PC 를 쓰지 않는다
set -uo pipefail
cd "$(dirname "$0")"

# 비밀번호가 필요한 일(DFC 용 시스템 패키지 설치)은 맨 먼저 한다. 자리를 비운 뒤에 멈춰 있지 않도록.
MISSING=""
for p in graphviz libgraphviz-dev build-essential python3-dev; do
  dpkg -s "$p" >/dev/null 2>&1 || MISSING="$MISSING $p"
done
if [ -n "$MISSING" ]; then
  echo "#### 먼저 시스템 패키지를 설치합니다:$MISSING"
  echo "#### WSL 비밀번호를 물으면 입력하세요 (입력해도 화면에 표시되지 않습니다)"
  if sudo apt-get update && sudo apt-get install -y $MISSING; then
    echo "#### 시스템 패키지 설치 완료. 이제 자리를 비워도 됩니다."
  else
    echo "#### 시스템 패키지 설치 실패. 파일럿과 탐침은 계속하고, DFC 단계는 건너뜁니다."
    SKIP_DFC=1
  fi
else
  echo "#### 시스템 패키지는 이미 있습니다. 비밀번호 입력 없이 끝까지 돕니다."
fi

if [ -f results/first_table.md ]; then
  echo "#### 1/3 파일럿: 이미 끝남 (results/first_table.md), 건너뜀"
else
  echo "#### 1/3 파일럿"
  bash setup_and_run.sh pilot || { echo "#### 파일럿 실패. 여기서 멈춥니다."; exit 1; }
fi

if [ -f results/all_runs.md ]; then
  echo "#### 2/3 탐침: 이미 끝남 (results/all_runs.md), 건너뜀"
else
  echo "#### 2/3 회복 탐침 + 캘리브레이션 A/B"
  bash setup_and_run.sh probes || echo "#### 탐침 실패 (파일럿 결과는 남아 있음). DFC 로 넘어갑니다."
fi

if [ -n "${SKIP_DFC:-}" ]; then
  echo "#### 3/3 DFC: 시스템 패키지가 없어 건너뜀"
else
  echo "#### 3/3 DFC 설치 + 스모크 테스트"
  bash dfc/setup_and_smoke.sh smoke < /dev/null || echo "#### DFC 단계 실패. logs/dfc 의 로그를 확인하세요."
fi

echo
echo "#### 전체 종료: $(date '+%F %T')"
[ -f results/first_table.md ] && echo "  첫 비교표: results/first_table.md"
[ -f results/all_runs.md ] && echo "  전체 표: results/all_runs.md"
ls logs/dfc/smoke_*.json >/dev/null 2>&1 && echo "  DFC 로그: logs/dfc/"
