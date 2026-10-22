# halio project — CLIP ViT-B/16 이미지 인코더 INT8 양자화 (ORT + Hailo-10H)

2026 모바일공학전공 하반기 공모전 트랙 2 자유주제.

> **처음 보거나 새 세션에서 이어받을 때는 [docs/00_새세션_시작안내.md](docs/00_새세션_시작안내.md) 부터.**
> **다른 PC에서 재현:** [docs/06_재현가이드.md](docs/06_재현가이드.md) → `bash reproduce.sh` (GPU 불필요)
> 현재 상태: [docs/01_인수인계_현재상태.md](docs/01_인수인계_현재상태.md) · 실험 노트: [docs/02_실험기록_전체.md](docs/02_실험기록_전체.md) · 측정 규칙: [docs/PROTOCOL.md](docs/PROTOCOL.md)

## 폴더 구조 (한눈에)

| 폴더·파일 | 내용 |
|---|---|
| `00_작업대기소_켜기.bat` | 켜 두면 `jobs/*.sh` 를 하나씩 WSL에서 자동 실행 (실험은 주로 이걸로 돌림) |
| `98_폴더정리.bat` / `99_불필요파일_삭제.bat` | 흩어진 파일 정리(옮기기만) / 삭제 후보를 확인 후 휴지통으로 |
| `실행파일_bat/` | 더블클릭 실행 파일. 1-x 설치, 2-x ORT 실험, 3-x Hailo 컴파일 |
| `docs/` | 문서: 시작 안내, 인수인계, 실험 기록, 프로토콜 |
| `src/` · `dfc/` · `tests/` | ORT 실험 코드 · Hailo DFC 코드 · 셀프테스트 |
| `splits/` | 검증셋·캘리브·COCO 검색 분할 목록 (고정) |
| `results/` · `logs/` · `jobs/` | 결과 · 로그 · 작업 대기소 (각 폴더의 README.md 참고) |
| `installers/` | Hailo DFC·Model Zoo 휠 (재배포 금지) |
| `tools/` · `_archive/` | 정리 스크립트 · 예전 백업 (실행에 안 씀) |
| 루트 `*.sh` | 실행 순서 스크립트. 서로 상대 경로로 부르므로 옮기지 않는다 |

## 더블클릭 실행 (`실행파일_bat/`)

| 파일 | 하는 일 |
|---|---|
| `1-1_WSL_배포판_확인.bat` | 설치된 WSL 배포판 목록을 `logs/wsl_list.txt` 에 저장 |
| `1-2_파이썬환경_설치_셀프테스트.bat` | 파이썬 환경 설치 + 다운로드 없는 점검 (약 5분) |
| `1-3_DFC용_시스템패키지_설치.bat` | DFC에 필요한 시스템 패키지를 WSL root로 설치 (비밀번호 불필요) |
| `1-4_DFC_설치_스모크테스트.bat` | Hailo DFC 설치 + 스모크 테스트 (2-1을 먼저) |
| `2-1_파일럿_첫비교표_A0_A1_A2.bat` | 정식 실행 → `results/first_table.md`. **다른 프로그램을 끄고 실행** |
| `2-2_회복탐침_캘리브AB.bat` | 회복 탐침 + 캘리브레이션 A/B |
| `2-3_정적INT8_A2m_재측정.bat` | 정적 INT8을 가중치 MatMul로 제한해 재측정 (A2m) |
| `2-4_레이어민감도_1단계.bat` | 레이어 민감도 1단계 (8개 구성) |
| `2-9_파일럿_탐침_DFC_한번에.bat` | 파일럿 → 탐침 → DFC 스모크를 차례로 |
| `3-1_HEF컴파일_공식설정.bat` | 1-4에 더해 Hailo-10H용 HEF 컴파일 |
| `3-2_HEF컴파일_CIFAR캘리브.bat` | CIFAR 캘리브레이션으로 HEF 컴파일 |

10/6 이후 실험(민감도 2·3단계, Hailo fc2 16비트, COCO 검색 등)은 bat 대신 작업 대기소로 돌렸다. 무엇을 돌렸는지는 `jobs/done/`, 출력은 `jobs/logs/`.

화면 내용은 `logs/run_<시각>_<모드>.log` 에도 남는다. 오류가 나면 이 로그를 보면 된다.
- 시스템 파이썬이 3.10~3.12 가 아니면(예: Ubuntu 26.04 의 3.14) uv 로 파이썬 3.10 을 사용자 홈에 받아 전용 환경을 만든다. sudo 가 필요 없다
- 기본 배포판이 아닌 WSL 배포판에서 돌리려면 이 폴더의 `wsl_distro.txt` 에 배포판 이름 한 줄(예: `Ubuntu-22.04`)을 적는다

아래는 같은 일을 WSL 터미널에서 직접 하는 방법이다.

## 처음 한 번: 환경 만들기 (PC00 의 WSL 에서)

```bash
cd "/mnt/c/Users/PC00/Desktop/dongchan/halio project"

# 파이썬 환경은 WSL 홈에 만든다 (/mnt/c 아래에 만들면 매우 느리다)
python3 -m venv ~/venvs/halio          # 안 되면: sudo apt install python3-venv
source ~/venvs/halio/bin/activate
pip install --upgrade pip

# PyTorch 는 CPU 전용판 (약 200MB). 기본판은 GPU 라이브러리까지 6GB 이상 받는다
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip freeze > requirements.lock.txt     # 실제 설치된 버전을 고정해 둔다 (재현용, 저장소에 커밋)
```

## 먼저: 다운로드 없이 코드만 검증 (약 2~3분)

```bash
SELFTEST=1 bash run_pilot.sh
```

가짜 데이터와 무작위 가중치로 전 과정을 돈다. 임시 폴더(`/tmp/halio_selftest.*`) 안에서만 돌아서 이 폴더의
결과나 split 을 건드리지 않는다. 끝에 표가 나오면 환경은 정상이다. **이 표의 수치는 무의미하다.**

## 정식 실행

```bash
bash run_pilot.sh
```

순서: 데이터 다운로드 → 검증셋 분리 → ONNX export 와 P1 게이트 → 양자화(A1, A2) → 정확도 평가
→ 연산자 커버리지 → 속도·메모리 측정 → `results/first_table.md`.

- 처음 실행 때 CLIP 가중치(약 350MB), CIFAR-100(약 170MB), COCO val2017(약 780MB)을 내려받는다
- **속도·메모리 측정(7단계) 동안에는 다른 프로그램을 끈다**
- 단계마다 산출물이 있으면 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 한다
- P1 게이트가 실패하면 거기서 멈춘다. 통과 전에는 다음 단계로 가지 않는다

첫 비교표에서 의미 있는 정확도 손실이 보이면 탐침을 돌린다.

```bash
bash run_probes.sh          # 회복 탐침 7종 + 캘리브레이션 도메인 A/B 5종 → results/all_runs.md
```

## Hailo DFC 스모크 테스트 (P5)

`hailo_dataflow_compiler-5.4.0-*.whl` 과 `hailo_model_zoo-5.4.0-*.whl` 을 `installers/` 폴더에 둔 상태에서
`실행파일_bat/1-4_DFC_설치_스모크테스트.bat` 을 실행한다 (또는 WSL 에서 `bash dfc/setup_and_smoke.sh`).

- DFC 전용 환경 `~/venvs/hailo` 를 따로 만든다. 파일럿 환경과 합치지 않는다 (numpy 1.26 / onnx 1.17 / tensorflow 2.19 로 버전이 다르다)
- 순서: Hailo 공식 CLIP ViT-B/16 ONNX 다운로드 → parse → 공식 모델 스크립트 적용 → COCO 64장으로 optimize → 에뮬레이터로 300장 간이 평가
- 모델 스크립트는 Model Zoo 휠 안의 공식 설정을 그대로 읽어 쓴다 (`optimization_level=0`, `calibset_size=64`). Model Zoo 패키지는 설치하지 않는다
- GPU 를 켠 상태로 실패하면 CPU 전용으로 한 번 더 시도한다
- 단계별 소요 시간과 성공 여부는 `logs/dfc/smoke_<시각>.json`. 실패해도 지우지 않는다
- 휠 파일은 Hailo 계정으로 받은 것이라 저장소에 올리지 않는다 (`.gitignore`)

10/5~14 동안 실제 DFC 5.4.0 으로 parse·optimize·에뮬레이터 평가·Hailo-10H 컴파일까지 돌아가는 것을 확인했다.

## 큰 파일은 D 드라이브에 둔다

C 드라이브 여유가 적어서 데이터·모델·가중치 캐시는 `/mnt/d/halio_data` (= `D:\halio_data`) 에 저장한다.
D 드라이브가 없으면 프로젝트 폴더에 저장한다. 다른 곳을 쓰려면 `export HALIO_DATA_DIR=/경로`.
코드·결과·로그·split 은 항상 이 폴더에 남는다.

## 고정된 측정 값 (바꾸면 그 전 수치를 전부 다시 재야 한다)

| 항목 | 값 | 위치 |
|---|---|---|
| 검증셋 크기 | 10,000장 (클래스당 100, 10/5 확정) | `src/common.py` 의 `VAL_PER_CLASS` |
| 측정 스레드 수 | 4 (코어 고정 없음) | `THREADS` 환경변수, 기본 4 |
| 비열등 마진 | COCO-select R@1 CI 하한 ≥ −2.0%p (10/14 확정) | `docs/PROTOCOL.md` |

## 파일

| 파일 | 역할 | 노션 작업 |
|---|---|---|
| `src/common.py` | **전처리·데이터 로딩·경로·CSV 스키마의 단일 소스** | 평가 프로토콜 고정 |
| `src/download_data.py` | CIFAR-100, COCO val2017 다운로드 | 데이터셋 확보 |
| `src/split_val.py` | 검증셋 분리, 캘리브레이션 목록, 겹침 0 확인 | 검증셋 분리 |
| `src/export_onnx.py` | ONNX export, P1 게이트, 텍스트 임베딩 사전 계산 | ONNX export, PyTorch↔ONNX 일치 |
| `src/backends.py` | 백엔드 추상화 (ORT 구현, DFC·HailoRT 는 자리만) | 공통 인터페이스 설계 |
| `src/evaluate.py` | 정확도, FP32 대비 일치율, 짝지은 비교 | FP32 기준점 |
| `src/quantize_ort.py` | 동적·정적 양자화, 설정 전부 기록 | 동적 INT8, 정적 INT8, 회복 탐침 |
| `src/coverage.py` | 연산자 커버리지, 프로파일링 | 연산자 커버리지 |
| `src/benchmark.py`, `src/bench_child.py` | 속도·메모리 (별도 프로세스) | 속도·메모리 프로토콜 |
| `src/make_table.py` | 비교표 | ★ 첫 비교표 |
| `run_pilot.sh`, `run_probes.sh` | 실행 순서 | — |
| `setup_and_run.sh`, `*.bat` | 환경 설치부터 실행까지 한 번에 (더블클릭용) | — |
| `dfc/setup_and_smoke.sh`, `dfc/smoke_test.py` | DFC 설치와 스모크 테스트 | Hailo DFC 설치, DFC 스모크 테스트 |
| `tests/make_fake_data.py` | 코드 검증용 가짜 데이터 | — |
| `src/sensitivity.py`, `run_sensitivity.sh` | 레이어 민감도 (층을 FP32로 남겨 회복량 측정) | 층별 민감도 |
| `src/act_stats.py` | 행렬곱 입력 활성값 분포 (이상치 확인) | 원인 규명 |
| `src/quantize_int4.py` | 가중치 전용 INT4 (MatMulNBits) | 개선 실험 |
| `src/split_outliers.py` | fc2 이상치 채널을 FP32 작은 행렬곱으로 분리 | 개선 실험 |
| `src/coco_retrieval.py` | COCO 텍스트→이미지 검색 평가 (분할·텍스트 임베딩·평가·표) | 주 평가 |
| `dfc/coco_ret_eval.py`, `dfc/compare_preds.py` | Hailo 에뮬레이터 검색 평가, Hailo 예측 짝지은 비교 | Hailo 이식 |
| `job_runner.sh` | 작업 대기소 (`00_작업대기소_켜기.bat`) | — |
| `tools/organize.sh`, `tools/delete_list.txt` | 폴더 정리, 삭제 후보 목록 | — |

## 결과물

```
results/first_table.md        첫 비교표
results/all_runs.md           탐침·A/B 포함 전체
results/accuracy.csv          정확도 (고정 스키마)
results/latency.csv           속도·메모리, 반복별
results/latency_summary.json  구성별 중앙값·표준편차
results/coverage.md / .json   연산자 커버리지
results/preds/                이미지별 예측·임베딩 (짝지은 비교용)
results/env_<hash>.json       측정 환경
logs/p1_parity.json           P1 게이트 로그
<D:\halio_data>/models/*.json 모델마다 양자화 설정 기록
```

## 검증된 환경

코드는 아래 두 환경에서 가짜 데이터로 끝까지 도는 것을 확인했다(2026-10-05). 실제 실험은 PC00 의 파이썬 3.10 환경(아래 둘째 줄)에서 돌렸다.

| 파이썬 | torch | onnx | onnxruntime | numpy |
|---|---|---|---|---|
| 3.13 | 2.14.1 | 1.23.1 | 1.30.0 | 2.5.3 |
| 3.10 | 2.14.1 | 1.23.1 | 1.23.2 | 2.2.6 |

## 알아둘 것

- 예선 비교표에 Hailo 공식 수치는 쓰지 않는다. ORT 결과를 Hailo 개선량으로 옮겨 말하지 않는다
- test 10,000장은 최종 구성 확정 전까지 열지 않는다. `evaluate.py` 는 `--eval-set test --i-know-final` 을 둘 다 줘야 연다
- 데이터 원본과 모델 파일은 저장소에 넣지 않는다 (`.gitignore`)
- Percentile·Entropy 캘리브레이션은 ORT 가 중간 텐서를 전부 메모리에 모으는 문제가 있어 2장씩 나눠 집계하도록 감쌌다 (`quantize_ort.patch_histogram_chunking`)
