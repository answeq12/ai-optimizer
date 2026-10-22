# 보고서 빈칸 채우기용 분석 3개 (10/18 23:20). 정확도·그래프 분석만, 속도 측정 없음 → PC 조용할 필요 없음. 약 10~20분
#  1) 파라미터 수·GFLOPs·파일 크기 (보고서 2장)        -> results/model_stats.md
#  2) 입력 종류별 양자화 손실, 사후 분석 (보고서 6.6)   -> results/failure_analysis_confirm.md, _select.md
#  3) Hailo HAR/HEF 에서 CPU 로 넘어가는 층 확인 (3.3) -> logs/dfc/hef_check.md
source ~/venvs/halio/bin/activate
python -m src.model_stats || echo "✘ model_stats"
python -m src.failure_analysis --split confirm > /dev/null || echo "✘ failure_analysis confirm"
python -m src.failure_analysis --split select > /dev/null || echo "✘ failure_analysis select"
grep -E "^## |캡션: 전체|숫자|글자|객체 16|< 사진의 10%" results/failure_analysis_confirm.md | head -40
deactivate
export CUDA_VISIBLE_DEVICES=''
source ~/venvs/hailo/bin/activate
ls -la /mnt/d/halio_data/hailo/*.compiled.har /mnt/d/halio_data/hailo/*.hef 2>/dev/null | awk '{print $5, $9}'
python -m dfc.hef_check || echo "✘ hef_check"
