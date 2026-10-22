# setup_and_run.sh 와 dfc/setup_and_smoke.sh 가 함께 쓰는 함수. 직접 실행하지 않는다.
#
# make_venv <경로>
#   파이썬 3.10~3.12 환경을 만든다. 시스템 파이썬이 그 범위이고 venv 모듈이 있으면 그것을 쓰고,
#   아니면(예: Ubuntu 26.04 의 파이썬 3.14, python3-venv 미설치) uv 로 파이썬 3.10 을 사용자 홈에 받아 만든다.
#   uv 방식은 sudo 가 필요 없고 시스템 파이썬을 건드리지 않는다.
#   3.10 으로 맞추는 이유: 이 코드를 검증한 버전이고, Hailo DFC 5.4.0 의 의존 패키지
#   (tensorflow 2.19.1, numpy 1.26.4, scipy 1.12.0)가 3.13 이상을 지원하지 않는다.
PY_TARGET=${HALIO_PYTHON:-3.10}

make_venv() {
  local venv=$1
  if [ -x "$venv/bin/python" ] && [ -x "$venv/bin/pip" ]; then
    return 0
  fi
  if [ -d "$venv" ]; then
    echo "== 만들다 만 환경을 지우고 다시 만듭니다: $venv"
    rm -rf "$venv"
  fi
  mkdir -p "$(dirname "$venv")"

  local sysver
  sysver=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none)
  case "$sysver" in
    3.10|3.11|3.12)
      if python3 -m venv "$venv" >/dev/null 2>&1 && [ -x "$venv/bin/pip" ]; then
        echo "== 파이썬 환경: 시스템 파이썬 $sysver 사용 ($venv)"
        return 0
      fi
      rm -rf "$venv"
      echo "== 시스템 파이썬 $sysver 에 venv 모듈이 없어 uv 로 만듭니다" ;;
    *)
      echo "== 시스템 파이썬이 $sysver 입니다. 검증된 $PY_TARGET 환경을 uv 로 따로 만듭니다 (시스템은 건드리지 않음)" ;;
  esac

  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    echo "== uv 설치 (사용자 홈 ~/.local/bin, sudo 불필요)"
    if command -v curl >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh || return 1
    elif command -v wget >/dev/null 2>&1; then
      wget -qO- https://astral.sh/uv/install.sh | sh || return 1
    else
      echo "!! curl 과 wget 이 모두 없습니다"; return 1
    fi
    export PATH="$HOME/.local/bin:$PATH"
  fi
  uv venv --python "$PY_TARGET" --seed "$venv" || return 1
  [ -x "$venv/bin/pip" ] || "$venv/bin/python" -m ensurepip --upgrade || return 1
  echo "== 파이썬 환경: uv 로 받은 $("$venv/bin/python" --version) ($venv)"
}
