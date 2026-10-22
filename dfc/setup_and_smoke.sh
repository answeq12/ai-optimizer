#!/usr/bin/env bash
# Hailo DFC 설치 + 스모크 테스트. Windows 에서 실행파일_bat/1-4_DFC_설치_스모크테스트.bat 을 더블클릭하면 WSL 에서 돈다.
#
#   bash dfc/setup_and_smoke.sh            # 설치 -> parse -> optimize -> 에뮬레이터 평가
#   bash dfc/setup_and_smoke.sh compile    # 위에 더해 HEF 컴파일까지
#
# DFC 는 파일럿 환경과 버전이 충돌하므로(numpy 1.26, onnx 1.17, tensorflow 2.19) 전용 환경 ~/venvs/hailo 를 쓴다.
# 화면 내용은 logs/dfc/setup_<시각>.log 에 남는다.
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-smoke}
VENV=${HAILO_VENV:-$HOME/venvs/hailo}
mkdir -p logs/dfc
LOG="logs/dfc/setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
fail() { echo; echo "!! 실패: $1"; echo "!! 로그: $LOG"; exit 1; }

echo "== DFC 설치·스모크 ($MODE) / $(date '+%F %T') / 로그: $LOG"
echo "== 시스템: $( (lsb_release -ds 2>/dev/null || grep -m1 PRETTY_NAME /etc/os-release) | tr -d '\"') / $(python3 --version 2>&1)"
echo "== 디스크 여유: 홈 $(df -h "$HOME" | awk 'NR==2{print $4}'), C 드라이브 $(df -h /mnt/c 2>/dev/null | awk 'NR==2{print $4}') / WSL 사용자: $(whoami)"
(command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader) || echo "== nvidia-smi 없음"

# 휠은 installers/ 에 둔다 (예전처럼 프로젝트 루트에 있어도 찾는다)
DFC_WHL=$(ls installers/hailo_dataflow_compiler-*.whl hailo_dataflow_compiler-*.whl 2>/dev/null | tail -1)
[ -n "$DFC_WHL" ] || fail "installers/ 폴더에 hailo_dataflow_compiler-*.whl 이 없습니다"
{ compgen -G "installers/hailo_model_zoo-*.whl" || compgen -G "hailo_model_zoo-*.whl"; } >/dev/null || fail "installers/ 폴더에 hailo_model_zoo-*.whl 이 없습니다"
echo "== DFC 휠: $DFC_WHL"

if [ ! -x "$VENV/bin/hailo" ]; then
  CHK=$HOME; [ -d /mnt/c ] && CHK=/mnt/c       # WSL 가상 디스크는 C 드라이브에 있다
  FREE_GB=$(df -BG --output=avail "$CHK" | tail -1 | tr -dc 0-9)
  [ "$FREE_GB" -ge 8 ] || fail "C 드라이브 여유가 ${FREE_GB}GB 입니다. DFC 환경 설치에 약 6~8GB 가 필요합니다. C 드라이브를 비운 뒤 다시 실행하세요."

  MISSING=""
  for p in graphviz libgraphviz-dev build-essential python3-dev; do
    dpkg -s "$p" >/dev/null 2>&1 || MISSING="$MISSING $p"
  done
  if [ -n "$MISSING" ]; then
    echo "== 시스템 패키지 설치가 필요합니다:$MISSING"
    echo "== WSL 비밀번호를 물으면 입력하세요 (화면에 표시되지 않습니다)"
    sudo apt-get update && sudo apt-get install -y $MISSING || fail "시스템 패키지 설치 실패. WSL 터미널에서 직접 실행하세요:
     sudo apt-get update && sudo apt-get install -y$MISSING"
  fi

  echo "== DFC 전용 파이썬 환경 만들기: $VENV"
  source ./lib_venv.sh
  make_venv "$VENV" || fail "파이썬 환경을 만들지 못했습니다"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || fail "파이썬 환경을 켤 수 없습니다: $VENV"
  pip install --upgrade pip || fail "pip 업그레이드 실패"
  # DFC 가 요구하는 torch 2.9.1 을 CPU 전용판으로 먼저 깐다 (기본판은 GPU 라이브러리까지 수 GB)
  pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu \
      || fail "torch 2.9.1 (CPU) 설치 실패"
  echo "== DFC 설치 (의존 패키지가 많아 10분 이상 걸릴 수 있습니다)"
  pip install "$DFC_WHL" || fail "DFC 설치 실패. 위 오류 메시지를 확인하세요"
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate" || fail "파이썬 환경을 켤 수 없습니다: $VENV"
fi

echo "== 설치 확인"
hailo --version || fail "hailo 명령이 동작하지 않습니다"
pip freeze > logs/dfc/requirements.hailo.lock.txt
python - <<'PY'
import numpy, onnx, onnxruntime, tensorflow as tf, torch
print(f"== 버전: tensorflow {tf.__version__}, torch {torch.__version__}, numpy {numpy.__version__}, "
      f"onnx {onnx.__version__}, onnxruntime {onnxruntime.__version__}")
PY

M=$(python -m src.common data-dir)
[ -d "$M/cifar-100-python" ] && [ -f splits/val.txt ] && [ -f splits/calib_coco_64_s0.txt ] \
  || fail "데이터와 split 이 없습니다. 먼저 실행파일_bat/2-1_파일럿_첫비교표_A0_A1_A2.bat (파일럿)을 실행하세요."

ARGS=("${@:2}"); [ "$MODE" = "compile" ] && ARGS=(--compile "${ARGS[@]}")   # 2번째 인자부터는 smoke_test 에 그대로 전달
echo "== 스모크 테스트 시작"
if ! python -m dfc.smoke_test "${ARGS[@]}"; then
  if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    echo
    echo "== 실패했습니다. GPU 를 끄고(CPU 전용) 한 번 더 시도합니다."
    echo "== (공식 설정은 optimization_level=0 이라 GPU 없이도 돌도록 되어 있습니다)"
    CUDA_VISIBLE_DEVICES='' python -m dfc.smoke_test "${ARGS[@]}" \
      || fail "CPU 전용으로도 실패. logs/dfc/smoke_*.json 의 failed_stage 를 확인하세요"
    echo "== 참고: GPU 를 켠 상태에서는 실패했고 CPU 전용으로 성공했습니다. 이 사실을 기록해 두세요."
  else
    fail "스모크 테스트 실패. logs/dfc/smoke_*.json 의 failed_stage 를 확인하세요"
  fi
fi
echo "== 완료. 로그: $LOG"
