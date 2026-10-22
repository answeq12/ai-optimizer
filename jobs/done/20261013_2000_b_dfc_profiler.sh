# 작업 2: DFC 프로파일러로 공식(13 컨텍스트) vs fc2 6·7 16비트(15 컨텍스트). 새 컴파일 없음. 추정치(실측 아님)
source ~/venvs/hailo/bin/activate
H=/mnt/d/halio_data/hailo/clip_vit_b_16_image_encoder
P=logs/dfc/profiler; mkdir -p $P
hailo profiler --help > $P/help.txt 2>&1; head -40 $P/help.txt
for t in official:$H.compiled.har fc2b67:$H.coco64_fc2b67.compiled.har; do
  name=${t%%:*}; har=${t#*:}
  echo "== $name: $har"; ls -la "$har" || continue
  ( cd $P && hailo profiler "$har" --out-path "$name.html" > "$name.log" 2>&1 \
     || hailo profiler "$har" > "$name.log2" 2>&1 ); echo "rc=$?"
  ls -la $P | grep -i "$name\|html\|json\|csv" || true
  tail -30 $P/$name.log $P/$name.log2 2>/dev/null
done
ls -la $P
