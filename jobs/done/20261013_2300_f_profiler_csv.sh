# 프로파일러: hailo10h 에서는 FPS·지연 추정 N/A → 컨텍스트별 자원 사용만 비교. CSV 를 구성별로 따로 저장
source ~/venvs/hailo/bin/activate
export CUDA_VISIBLE_DEVICES=''
H=/mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder; P=logs/dfc/profiler
for t in official:$H.compiled.har fc2b67:$H.coco64_fc2b67.compiled.har; do
  n=${t%%:*}; ( cd $P && hailo profiler "${t#*:}" --out-path "$n.html" --csv "$n.csv" > "$n.csv.log" 2>&1 ); echo "$n rc=$?"
done
ls -la $P/*.csv
