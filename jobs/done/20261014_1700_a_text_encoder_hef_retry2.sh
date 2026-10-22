# 재시도 2: forward hook 반환값 수정
# 텍스트 인코더(CLIP ViT-B/16) Hailo-10H HEF 컴파일 시도. Model Zoo 5.4.0 공식 설정. 부가 실험(예선 본 결과 아님)
# 1) 파일럿 환경: 공식 ONNX 다운로드, 구조 확인, 칩 밖 앞뒤 처리 검증, 캘리브(제외 목록 캡션) 준비
# 2) DFC 환경: parse(Add_1~Add_1515) -> optimize(공식 alls, finetune 실패 시 끄고 재시도) -> 간이 평가 -> compile
# 3) 프로파일러(추정치). 전체 4시간 상한
export CUDA_VISIBLE_DEVICES=''
source ~/venvs/halio/bin/activate
python -m src.text_hailo_prep || { echo "✘ 텍스트 준비 실패"; exit 1; }
deactivate
source ~/venvs/hailo/bin/activate
timeout 4h python -m dfc.text_encoder_compile; rc=$?
echo "== text_encoder_compile rc=$rc"
H=/mnt/d/halio_data/hailo/clip_vit_b_16_text_encoder
ls -la $H.* 2>/dev/null
if [ -f $H.compiled.har ]; then
  P=logs/dfc/profiler; mkdir -p $P
  ( cd $P && hailo profiler $H.compiled.har --out-path text.html --csv text.csv > text.log 2>&1 ); echo "profiler rc=$?"
  python3 - <<'PY'
import csv
try:
    rows=list(csv.reader(open("logs/dfc/profiler/text.csv")))
    d=dict(zip(rows[0],rows[1])); i=[k for k,r in enumerate(rows) if r and r[0]=="context_name"][0]; hh=rows[i]; t=0
    for r in rows[i+2:]:
        if not r or not r[0]: break
        t+=1/float(dict(zip(hh,r))["fps"])
    print(f"text: 컨텍스트 {d['number_of_contexts']}, 16비트 SC {d.get('scs_16bit')}, 추정 {1/t:.0f} fps (전환 오버헤드 제외, 추정치)")
except Exception as e:
    print("프로파일러 요약 실패:", e)
PY
fi
exit $rc
