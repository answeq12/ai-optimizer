# E46 준비: CC3M(webdataset) 파일 목록·크기 확인만 (모델 평가 없음)
source ~/venvs/halio/bin/activate
python - <<'PY'
import json, urllib.request
d = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/pixparse/cc3m-wds", timeout=60))
s = [x["rfilename"] for x in d.get("siblings", [])]
print("파일 수", len(s)); print("기타", [x for x in s if not x.endswith(".tar")])
for k in ("train", "valid"):
    t = sorted(x for x in s if k in x and x.endswith(".tar")); print(k, len(t), t[:2], t[-2:])
print("license", d.get("cardData", {}).get("license"))
PY
curl -sIL https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/cc3m-validation-0000.tar | grep -iE "^HTTP|content-length" | tail -2
curl -sL https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/README.md | head -60
