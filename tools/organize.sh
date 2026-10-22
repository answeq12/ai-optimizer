#!/usr/bin/env bash
# 폴더 정리 (몇 번 실행해도 같은 결과). 98_폴더정리.bat 이 WSL 에서 이 파일을 실행한다.
# 지우는 것은 없다. 옮기기만 한다. 삭제는 99_불필요파일_삭제.bat (휴지통으로 보냄).
#  1) 루트의 실험용 bat 을 실행파일_bat/ 로 옮기고 이름을 하는 일이 보이게 바꾼다
#  2) Hailo 설치 휠을 installers/ 로 옮긴다 (dfc 코드는 두 위치를 모두 찾는다)
#  3) DFC 가 실행 위치에 자동으로 남기는 로그를 logs/dfc/hailo_auto_logs/ 로 옮긴다
#  4) 코드 백업 파일(*.bak_*)을 _archive/code_backups/ 로 옮긴다 (원래 폴더 구조 유지)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
if [ -f jobs/running.txt ]; then
  echo "!! 작업 대기소에서 실험이 돌고 있습니다: $(cat jobs/running.txt)"
  echo "!! 실험이 끝난 뒤 다시 실행하세요. 아무것도 옮기지 않았습니다."
  exit 1
fi
n=0
mv_safe() {   # mv_safe 원본 대상 : 대상이 이미 있으면 건너뛴다
  [ -e "$1" ] || return 0
  if [ -e "$2" ]; then echo "  건너뜀(이미 있음): $2"; return 0; fi
  mkdir -p "$(dirname "$2")"
  mv "$1" "$2" && echo "  $1  ->  $2" && n=$((n+1))
}

echo "== 1) bat 정리"
B=실행파일_bat
move_bat() {  # 하위 폴더로 옮기면서 cd 경로를 상위 폴더로 고친다
  [ -e "$1" ] || return 0
  if [ -e "$B/$2" ]; then echo "  건너뜀(이미 있음): $B/$2"; return 0; fi
  mkdir -p "$B"
  sed 's#cd /d "%~dp0"#cd /d "%~dp0.."#' "$1" > "$B/$2" && rm -f "$1" 2>/dev/null
  if [ -e "$1" ]; then   # 삭제 권한이 없는 환경이면 원본을 보관 폴더로
    mv "$1" "_archive/old_bat/$1" 2>/dev/null || { mkdir -p _archive/old_bat; mv "$1" "_archive/old_bat/$1"; }
  fi
  echo "  $1  ->  $B/$2"; n=$((n+1))
}
mv_safe 0_JOB_RUNNER.bat                00_작업대기소_켜기.bat
move_bat 0_check_wsl.bat                1-1_WSL_배포판_확인.bat
move_bat 1_setup_and_selftest.bat       1-2_파이썬환경_설치_셀프테스트.bat
move_bat 4a_install_system_packages.bat 1-3_DFC용_시스템패키지_설치.bat
move_bat 4_dfc_setup_and_smoke.bat      1-4_DFC_설치_스모크테스트.bat
move_bat 2_run_pilot.bat                2-1_파일럿_첫비교표_A0_A1_A2.bat
move_bat 3_run_probes.bat               2-2_회복탐침_캘리브AB.bat
move_bat 6_run_a2fix.bat                2-3_정적INT8_A2m_재측정.bat
move_bat 8_run_sensitivity.bat          2-4_레이어민감도_1단계.bat
move_bat 9_run_everything.bat           2-9_파일럿_탐침_DFC_한번에.bat
move_bat 5_dfc_compile_hef.bat          3-1_HEF컴파일_공식설정.bat
move_bat 7_dfc_compile_cifar_calib.bat  3-2_HEF컴파일_CIFAR캘리브.bat

echo "== 2) Hailo 설치 휠 -> installers/"
for f in hailo_dataflow_compiler-*.whl hailo_model_zoo-*.whl; do
  [ -e "$f" ] && mv_safe "$f" "installers/$f"
done

echo "== 3) DFC 자동 로그 -> logs/dfc/hailo_auto_logs/"
for f in hailo_sdk.*.log acceleras.log allocator.log system_reqs_results.log dfc/hailo_sdk.*.log; do
  [ -e "$f" ] || continue
  d="logs/dfc/hailo_auto_logs/$(basename "$f")"
  if [ -e "$d" ]; then d="logs/dfc/hailo_auto_logs/$(date +%Y%m%d_%H%M%S)_$(basename "$f")"; fi
  mv "$f" "$d" 2>/dev/null || { mkdir -p logs/dfc/hailo_auto_logs; mv "$f" "$d"; }
  echo "  $f  ->  $d"; n=$((n+1))
done

echo "== 4) 코드 백업(*.bak_*) -> _archive/code_backups/"
while IFS= read -r f; do
  f=${f#./}
  mv_safe "$f" "_archive/code_backups/$f"
done < <(find . -path ./_archive -prune -o -path ./jobs -prune -o -name '*.bak_*' -type f -print)

echo
echo "== 끝. 옮긴 항목 $n 개. 삭제한 것은 없습니다."
