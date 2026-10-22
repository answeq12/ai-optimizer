# 작업 4: 블록 7 fc2만 a16_w16 (공식 설정 + COCO 64 s0). 6·7 둘 다(15 컨텍스트, 추정 140fps) 대비 속도·정확도 절충 확인. 약 1~1.5시간
export CUDA_VISIBLE_DEVICES=''
H=/mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder
bash dfc/setup_and_smoke.sh compile --calib-list splits/calib_coco_64_s0.txt --tag coco64_fc2b7 \
  --fc2-16bit-blocks 7 --eval-n 300
source ~/venvs/hailo/bin/activate
python -m dfc.coco_ret_eval --har $H.coco64_fc2b7.quantized.har --code H-fc2b7
python -m src.coco_retrieval table --ref H-FP --codes H-FP H-official H-fc2b7 H-fc2b67 --margin 2.0 --tag hailo
P=logs/dfc/profiler
( cd $P && hailo profiler $H.coco64_fc2b7.compiled.har --out-path fc2b7.html --csv fc2b7.csv > fc2b7.log 2>&1 ); echo "profiler rc=$?"
python3 - <<'PY'
import csv
for n in ["official","fc2b7","fc2b67"]:
    try: rows=list(csv.reader(open(f"logs/dfc/profiler/{n}.csv")))
    except FileNotFoundError: print(n,"없음"); continue
    d=dict(zip(rows[0],rows[1])); i=[k for k,r in enumerate(rows) if r and r[0]=="context_name"][0]; hh=rows[i]; t=0
    for r in rows[i+2:]:
        if not r or not r[0]: break
        t+=1/float(dict(zip(hh,r))["fps"])
    print(f"{n}: 컨텍스트 {d['number_of_contexts']}, 16비트 SC {d['scs_16bit']}, 추정 {1/t:.0f} fps (전환 오버헤드 제외, 추정치)")
PY
