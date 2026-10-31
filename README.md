# CLIP ViT-B/16 이미지 인코더 INT8 경량화 (ONNX Runtime + Hailo-10H)

2026 모바일공학전공 하반기 공모전 트랙 2 자유주제, 팀 PCA.
휴대폰 갤러리의 일상 사진을 영어 문장으로 찾는 기기 안 사진 검색을 위해, CLIP ViT-B/16 이미지 인코더를 정적 INT8로 경량화했다
(이상치 채널 분리 SPL + 뒤쪽 4블록 도메인 적응).

> **재현 방법: [docs/06_재현가이드.md](docs/06_재현가이드.md)** → `bash reproduce.sh` (GPU 불필요, 2~4시간)
> 제출 모델 파일: [model_files/](model_files/) · 적응 가중치: [Release v1.0-da5](https://github.com/answeq12/ai-optimizer/releases/tag/v1.0-da5)

## 폴더 구조

이 저장소에는 보고서 9장의 재현에 필요한 파일만 있다.

| 폴더·파일 | 내용 |
|---|---|
| `reproduce.sh` | 전체 재현: 다운로드 → 분할 재생성·sha256 비교 → FP32 export·출력 일치 검사 → 양자화 → 최종 구성 생성 → COCO-confirm·CIFAR-100 test → 속도 → 기대값 비교 (2~4시간) |
| `repro_quick.sh` | 빠른 재현: 소규모 서브셋으로 같은 과정 (20~40분) |
| `run_pilot.sh` | `SELFTEST=1 bash run_pilot.sh`: 다운로드 없이 가짜 데이터로 코드만 점검 (2~3분) |
| `repro/` | 최종 모델 만들기(`da_build.sh`), 기대값(`EXPECTED*.json`), 기대값 비교(`check_expected.py`), 분할·가중치 sha256 |
| `src/` | 코드 전부: export·양자화(`quantize_ort.py`, `split_outliers.py`)·평가(`coco_retrieval.py`, `evaluate.py`, `*_eval.py`)·속도(`benchmark.py`, `index_bench.py`)·학습(`da_*.py`) |
| `tests/` | SELFTEST용 가짜 데이터 |
| `splits/` | 고정된 평가·캘리브·학습 사진 목록 |
| `results/sensitivity/groups.json` | 양자화 단계가 읽는 층 이름 목록 (재현 결과는 실행 시 `results/`·`logs/`에 새로 생긴다) |
| `model_files/` | 제출 모델 파일: 베이스라인 FP32 ONNX(4조각), 최종 `DA5-SPL-b567-k32.onnx`, 적응 텍스트, 참고용 SPL ONNX·HEF |
| `dfc/` | Hailo-10H (선택, Hailo DFC 5.4.0 필요): 적응 가중치 넣기(`inject_da.py`), 이상치 갈래 분리·컴파일(`split_hailo.py`), 에뮬레이터 평가(`final_eval.py`) |
| `requirements.lock.txt` | 버전 고정 설치 목록 |

## 환경

WSL Ubuntu 22.04, Python 3.10, torch 2.14.1+cpu, onnxruntime 1.23.2, open_clip_torch 3.3.0, CPU 4스레드, 배치 1.
데이터·모델 등 큰 파일은 `HALIO_DATA_DIR`(기본: `/mnt/d` 가 있으면 `/mnt/d/halio_data`, 없으면 이 폴더)에 저장된다.
