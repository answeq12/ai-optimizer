# 텍스트 인코더 HEF: finetune 끈 버전 (공식 alls 에서 finetune 한 줄만 disabled). 10/14 18:20 투입
# 이유: 공식 설정의 finetune 1에폭이 CPU 에서 80분을 넘겨 4에폭이 4시간 상한 안에 못 끝남. 공식 설정과 다르다는 점을 기록
# 준비(prep.npz)는 앞 작업에서 끝남
export CUDA_VISIBLE_DEVICES=''
source ~/venvs/hailo/bin/activate
timeout -k 5m 2h python -m dfc.text_encoder_compile --no-finetune --tag noft; rc=$?
echo "== text_encoder_compile(noft) rc=$rc"
H=/mnt/d/halio_data/hailo/clip_vit_b_16_text_encoder.noft
ls -la $H.* 2>/dev/null
if [ -f $H.compiled.har ]; then
  P=logs/dfc/profiler; mkdir -p $P
  ( cd $P && hailo profiler $H.compiled.har --out-path text_noft.html --csv text_noft.csv > text_noft.log 2>&1 ); echo "profiler rc=$?"
  python3 - <<'PY'
import csv
try:
    rows=list(csv.reader(open("logs/dfc/profiler/text_noft.csv")))
    d=dict(zip(rows[0],rows[1])); i=[k for k,r in enumerate(rows) if r and r[0]=="context_name"][0]; hh=rows[i]; t=0
    for r in rows[i+2:]:
        if not r or not r[0]: break
        t+=1/float(dict(zip(hh,r))["fps"])
    print(f"text_noft: 컨텍스트 {d['number_of_contexts']}, 16비트 SC {d.get('scs_16bit')}, 추정 {1/t:.0f} fps (전환 오버헤드 제외, 추정치)")
except Exception as e:
    print("프로파일러 요약 실패:", e)
PY
fi
# (10/15 수정) STOP 생성 삭제: WSL 재시작 후 창이 하나뿐이고, 뒤에 새벽 측정 작업이 이어져야 함
exit $rc
