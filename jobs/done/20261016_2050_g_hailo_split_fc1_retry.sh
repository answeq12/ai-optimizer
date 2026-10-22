# Hailo 이상치 채널 격리 3차 (10/16 20:45 투입): fc1+GELU 까지 rest/hot 으로 분할(--split-fc1) → GELU 출력 hot 32채널이 처음부터 16비트
# 2차 결과: fc2 만 나누면 보호 블록을 늘려도 −1.3~−1.6 고정 (conv(fc1+GELU) 출력 3072채널이 8비트로 먼저 양자화되는 것이 원인이라는 가설)
# 밤 속도 측정(01:00~)과 겹치지 않게 00:30 에 강제 종료
export CUDA_VISIBLE_DEVICES=''
source ~/venvs/hailo/bin/activate
END=$(date -d "00:30 tomorrow" +%s); [ $(date +%H) -lt 1 ] && END=$(date -d "00:30" +%s)
left() { echo $(( END - $(date +%s) )); }
H=/mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder
P=logs/dfc/profiler; mkdir -p $P
CODES="H-FP H-official H-fc2b7 H-fc2b67 H-spl-b67-k32 H-spl-b567-k32"
for cfg in "7 splf_b7_k32 32" "6,7 splf_b67_k32 32"; do
  set -- $cfg; B=$1; T=$2; K=$3; CODE=H-${T//_/-}
  [ $(left) -gt 1800 ] || { echo "시간 부족, $T 건너뜀"; continue; }
  echo "== $T (블록 $B) $(date '+%T')"
  timeout -k 5m $(left) python -m dfc.split_hailo --blocks $B --k $K --tag $T --split-fc1 || { echo "✘ $T 실패 (logs/dfc/split_$T.json)"; continue; }
  timeout -k 5m $(left) python -m dfc.coco_ret_eval --har $H.$T.quantized.har --code $CODE && CODES="$CODES $CODE"
  ( cd $P && timeout 20m hailo profiler $H.$T.compiled.har --out-path $T.html --csv $T.csv > $T.log 2>&1 ); echo "profiler rc=$?"
done
python -m src.coco_retrieval table --ref H-FP --codes $CODES --margin 2.0 --tag hailo_spl3
cat results/coco_ret_select_hailo_spl3.md
python3 - <<'PY'
import csv
for n in ["official","fc2b7","fc2b67","spl_b7_k32","spl_b67_k32","spl_b567_k32","splf_b7_k32","splf_b67_k32"]:
    try: rows=list(csv.reader(open(f"logs/dfc/profiler/{n}.csv")))
    except FileNotFoundError: print(n,"없음"); continue
    d=dict(zip(rows[0],rows[1])); i=[k for k,r in enumerate(rows) if r and r[0]=="context_name"][0]; hh=rows[i]; t=0
    for r in rows[i+2:]:
        if not r or not r[0]: break
        t+=1/float(dict(zip(hh,r))["fps"])
    print(f"{n}: 컨텍스트 {d['number_of_contexts']}, 16비트 SC {d.get('scs_16bit')}, 추정 {1/t:.0f} fps (추정치)")
PY
echo "== 끝 $(date '+%F %T')"
