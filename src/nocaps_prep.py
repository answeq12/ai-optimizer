"""E34 최종 테스트셋 준비: NoCaps val 4,500장 + 캡션. 이미지 평가는 하지 않는다(봉인).

  ~/venvs/data/bin/python src/nocaps_prep.py <저장 폴더>

- 사진·캡션: Hugging Face lmms-lab/NoCaps (validation, parquet). 사진 바이트를 다시 인코딩하지 않고 그대로 저장
- 묶음(in/near/out-of-domain): 공식 nocaps_val_4500_captions.json 에 domain 정보가 있으면 함께 저장
- 표준 라이브러리 + pyarrow + huggingface_hub 만 쓴다 (별도 venv: 메인 재현 환경을 건드리지 않음)
"""
import json, sys, urllib.request
from pathlib import Path
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

out = Path(sys.argv[1]); (out / "images").mkdir(parents=True, exist_ok=True)
repo = "lmms-lab/NoCaps"
files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet") and "val" in f.lower()]
print("[nocaps] parquet:", files, flush=True)
caps, meta = {}, {}
for f in files:
    t = pq.read_table(hf_hub_download(repo, f, repo_type="dataset"))
    print("[nocaps] 열:", t.column_names, t.num_rows, flush=True)
    for r in t.to_pylist():
        fn = r.get("image_file_name") or f"{r['image_id']}.jpg"
        (out / "images" / fn).write_bytes(r["image"]["bytes"])
        caps[fn] = [c.strip() for c in r["annotations_captions"]]
        meta[fn] = {"image_id": r.get("image_id"), "open_images_id": r.get("image_open_images_id"),
                    "w": r.get("image_width"), "h": r.get("image_height")}
dom = {}
for url in ("https://s3.amazonaws.com/nocaps/nocaps_val_4500_captions.json",
            "https://nocaps.s3.amazonaws.com/nocaps_val_4500_captions.json"):
    try:
        j = json.loads(urllib.request.urlopen(url, timeout=60).read())
        for im in j["images"]:
            if "domain" in im: dom[im["file_name"]] = im["domain"]
        print(f"[nocaps] 공식 주석: 사진 {len(j['images'])}, domain 표시 {len(dom)}", flush=True); break
    except Exception as e:
        print(f"[nocaps] 공식 주석 실패 {url}: {e!r}", flush=True)
for fn in meta: meta[fn]["domain"] = dom.get(fn)
(out / "captions.json").write_text(json.dumps({"source": repo, "captions": caps, "meta": meta}, ensure_ascii=False))
from collections import Counter
print(f"[nocaps] 사진 {len(caps)}, 캡션 {sum(map(len, caps.values()))}, 묶음 {Counter(m['domain'] for m in meta.values())}")
