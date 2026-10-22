# jobs/ 안내 — 작업 대기소

`00_작업대기소_켜기.bat` 창을 켜 두면 `job_runner.sh` 가 15초마다 이 폴더를 보고 `*.sh` 를 이름 순서대로 하나씩 WSL에서 실행한다.

| 위치 | 내용 |
|---|---|
| `*.sh` | 대기 중인 작업 |
| `running.txt` | 지금 도는 작업 (없으면 쉬는 중) |
| `.heartbeat` | 마지막 확인 시각(epoch 초). 오래됐으면 대기소가 꺼진 것 |
| `done/` | 실행한 작업 스크립트 (지우지 않는다. 무엇을 돌렸는지의 기록) |
| `logs/<이름>.log` | 작업 출력. 마지막 줄 `rc=<종료코드>` |
| `cancelled/` | 넣었다가 취소한 작업 |
| `STOP` | 이 파일을 만들면 지금 작업을 마친 뒤 대기소 종료 |

작업 파일 이름: `YYYYMMDD_HHMM_<순서>_<이름>.sh`. 작업 폴더는 프로젝트 루트.

- 파일럿 환경: 첫 줄 `source ~/venvs/halio/bin/activate`
- DFC: `export CUDA_VISIBLE_DEVICES=''` 후 `bash dfc/setup_and_smoke.sh ...`
- `set -e` 가 없으면 중간 실패해도 rc=0 이 찍힌다. 로그에서 `✘|실패|Error` 확인
- 속도 측정 작업은 다른 프로그램이 없을 때만 넣는다
