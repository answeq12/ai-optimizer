# logs/ 안내

| 위치 | 내용 |
|---|---|
| `run_<날짜>_<모드>.log` | bat/setup_and_run.sh 로 돌린 실행 로그 (selftest, pilot, probes, a2fix, sens…) |
| `p1_parity.json` | PyTorch↔ONNX 일치 확인(P1 게이트) |
| `coco_ret_split_check.json` | COCO 검색 분할 sha256·겹침 확인 |
| `profile_<구성>.json` | ORT 프로파일 원본 (각 10~15MB). 요약은 results/coverage.md |
| `wsl_list.txt` | WSL 배포판 목록 |
| `dfc/setup_*.log`, `dfc/smoke_*.json` | DFC 실행 로그·단계별 결과 (태그: coco64_base, coco64_fc2b67 …) |
| `dfc/*.alls` | 실제로 쓴 Hailo 모델 스크립트 (구성별) |
| `dfc/layers.*.json` | 16비트로 지정한 Hailo 층 목록 |
| `dfc/preds.*.npz`, `dfc/compare_preds*.md` | Hailo 에뮬레이터 예측과 짝지은 비교 |
| `dfc/profiler/` | DFC 프로파일러 결과 (추정 fps, 컨텍스트 수) |
| `dfc/hailo_auto_logs/` | DFC가 실행 위치에 자동으로 남기는 로그 (98_폴더정리.bat 이 모음). 지워도 됨 |

작업 대기소로 돌린 실험의 출력은 `jobs/logs/` 에 있다.
