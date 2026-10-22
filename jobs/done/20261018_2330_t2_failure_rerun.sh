# failure_analysis 점추정을 캡션 단위로 고쳐 다시 (표의 두 R@1 차와 일치하게). 약 1분
source ~/venvs/halio/bin/activate
python -m src.failure_analysis --split confirm > /dev/null && python -m src.failure_analysis --split select > /dev/null && echo done || echo "✘ failure_analysis"
