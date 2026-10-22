# results/ 안내

코드가 정해진 이름으로 덮어쓰는 파일들이라 **이름을 바꾸지 않는다**. 구성 이름 뜻은 `docs/02_실험기록_전체.md` 참고.

| 파일 | 내용 | 만드는 코드 |
|---|---|---|
| `first_table.md` | 첫 비교표 A0·A1·A2 (10/5) | src/make_table.py |
| `all_runs.md` | **CIFAR 검증 전체 구성 표** (정확도·속도·크기) + McNemar | src/make_table.py |
| `accuracy.csv` | 구성별 정확도 원본 (고정 스키마) | src/evaluate.py |
| `latency.csv` | 10/5 첫 속도 측정 (반복별) | src/benchmark.py |
| `latency_a2fix.csv` | 10/6 A2m 속도 | src/benchmark.py |
| `latency_a3b.csv` | 10/6 혼합 정밀도 후보 속도 | src/benchmark.py |
| `latency_final.csv` | **10/13 최종 후보 속도 (보고서용 세션)** | src/benchmark.py |
| `latency_summary.json` | 속도 중앙값·표준편차 요약 | src/benchmark.py |
| `coverage.md` / `.json` | 연산자 커버리지 (정수 연산 비율, 정수 커널 시간) | src/coverage.py |
| `sensitivity.md`, `sensitivity/` | 레이어 민감도 결과, 구성별 제외 노드 목록 | src/sensitivity.py |
| `act_stats*.md` / `.json` | fc2 등 활성값 분포 (COCO 캘리브 / CIFAR / COCO-select) | src/act_stats.py |
| `coco_ret_select.md` | **COCO 검색 select 결과 (ORT)** | src/coco_retrieval.py |
| `coco_ret_select_hailo.md` | **COCO 검색 select 결과 (Hailo)** | src/coco_retrieval.py |
| `coco_ret/` | 검색 평가용 구성별 이미지 임베딩·성적 (*-select.npz/json) | src/coco_retrieval.py, dfc/coco_ret_eval.py |
| `preds/` | CIFAR 이미지별 예측·임베딩 (*-val.npz/json). 짝지은 비교에 필요, 지우지 않는다 | src/evaluate.py |
| `env_<hash>.json` | 측정 환경 기록 | src/common.py |

Hailo CIFAR 비교 표는 `logs/dfc/compare_preds.md`(2,000장), `compare_preds_n5000.md`(5,000장).
