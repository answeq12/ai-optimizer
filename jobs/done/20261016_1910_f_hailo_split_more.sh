# Hailo 이상치 채널 격리 2차 (10/16 19:10 투입). 1차(E21): b7/b67 k32 → 컨텍스트 13 회복, R@1 −1.43/−1.27 로 마진 경계 미달
# 2차: 보호 범위 확대(b5·6·7, k32) / hot 채널 확대(b7, k64). 비교 대상 H-fc2b7(−0.48, 14 컨텍스트)
# 밤 속도 측정(01:00~)과 겹치지 않게 00:30 에 강제 종료
export CUDA_VISIBLE_DEVICES=''
source ~/venvs/hailo/bin/activate
END=$(date -d "00:30 tomorrow" +%s); [ $(date +%H) -lt 1 ] && END=$(date -d "00:30" +%s)
left() { echo $(( END - $(date +%s) )); }
H=/mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder
P=logs/dfc/profiler; mkdir -p $P
CODES="H-FP H-official H-fc2b7 H-fc2b67 H-spl-b7-k32 H-spl-b67-k32"
for cfg in "5,6,7 spl_b567_k32 32" "7 spl_b7_k64 64"; do
  set -- $cfg; B=$1; T=$2; K=$3; CODE=H-${T//_/-}
  [ $(left) -gt 1800 ] || { echo "시간 부족, $T 건너뜀"; continue; }
  echo "== $T (블록 $B) $(date '+%T')"
  timeout -k 5m $(left) python -m dfc.split_hailo --blocks $B --k $K --tag $T || { echo "✘ $T 실패 (logs/dfc/split_$T.json)"; continue; }
  timeout -k 5m $(left) python -m dfc.coco_ret_eval --har $H.$T.quantized.har --code $CODE && CODES="$CODES $CODE"
  ( cd $P && timeout 20m hailo profiler $H.$T.compiled.har --out-path $T.html --csv $T.csv > $T.log 2>&1 ); echo "profiler rc=$?"
done
python -m src.coco_retrieval table --ref H-FP --codes $CODES --margin 2.0 --tag hailo_spl2
cat results/coco_ret_select_hailo_spl2.md
python3 - <<'PY'
import csv
for n in ["official","fc2b7","fc2b67","spl_b7_k32","spl_b67_k32","spl_b567_k32","spl_b7_k64"]:
    try: rows=list(csv.reader(open(f"logs/dfc/profiler/{n}.csv")))
    except FileNotFoundError: print(n,"없음"); continue
    d=dict(zip(rows[0],rows[1])); i=[k for k,r in enumerate(rows) if r and r[0]=="context_name"][0]; hh=rows[i]; t=0
    for r in rows[i+2:]:
        if not r or not r[0]: break
        t+=1/float(dict(zip(hh,r))["fps"])
    print(f"{n}: 컨텍스트 {d['number_of_contexts']}, 16비트 SC {d.get('scs_16bit')}, 추정 {1/t:.0f} fps (추정치)")
PY
echo "== 끝 $(date '+%F %T')"
