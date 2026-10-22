# 데이터셋 원본 사진 예시 복사 (1분): docs/samples/ 에 CIFAR-100(32x32 원본 + 224 확대), COCO val2017·train2017, Flickr30k test 각 몇 장 + 크기·캡션 정리
set -eo pipefail
source ~/venvs/halio/bin/activate
python - <<'PY'
import json, random
from pathlib import Path
from PIL import Image
from src import common as C
out=Path("docs/samples"); out.mkdir(parents=True, exist_ok=True); L=["# 데이터셋 원본 사진 예시", ""]
imgs,labels=C.load_cifar100("test"); names=C.cifar100_class_names()
for i in (0,1,2):
    Image.fromarray(imgs[i]).save(out/f"cifar100_test{i}_32px.png")
    Image.fromarray(C.preprocess_uint8(Image.fromarray(imgs[i]))).save(out/f"cifar100_test{i}_224px.png")
    L.append(f"- cifar100_test{i}: 원본 32x32, 정답 '{names[labels[i]]}' (모델 입력은 224x224 bicubic 확대)")
cap=json.loads((C.DATA/"coco/annotations/captions_val2017.json").read_text())
sel=[l.strip() for l in open("splits/coco_ret_confirm.txt").read().split("\n")[1:] if l.strip()][:3]
fid={im["file_name"]:im["id"] for im in cap["images"]}
for f in sel:
    im=Image.open(C.COCO_DIR/f); im.save(out/f"coco_val_{f}")
    cs=[a["caption"].strip() for a in cap["annotations"] if a["image_id"]==fid[f]]
    L.append(f"- coco_val_{f}: {im.size[0]}x{im.size[1]}, 평가(confirm). 캡션: " + " / ".join(cs))
tr=json.loads((C.DATA/"coco/annotations/captions_train2017.json").read_text()); tfid={im["file_name"]:im["id"] for im in tr["images"]}
for f in [l.strip() for l in open("splits/da_train_30k.txt").read().split("\n")[1:] if l.strip()][:2]:
    im=Image.open(C.DATA/"coco/train2017"/f); im.save(out/f"coco_train_{f}")
    cs=[a["caption"].strip() for a in tr["annotations"] if a["image_id"]==tfid[f]][:2]
    L.append(f"- coco_train_{f}: {im.size[0]}x{im.size[1]}, 학습용. 캡션 예: " + " / ".join(cs))
fl=json.loads((C.DATA/"flickr1k/captions.json").read_text())["captions"]
fs=sorted(fl); sizes=[Image.open(C.DATA/"flickr1k/images"/f).size for f in fs]
for f in fs[:3]:
    im=Image.open(C.DATA/"flickr1k/images"/f); im.save(out/f"flickr_{f}")
    L.append(f"- flickr_{f}: {im.size[0]}x{im.size[1]}, 평가(test). 캡션: " + " / ".join(fl[f][:5]))
import collections
L+=["", f"Flickr30k 1K 크기 분포: 긴 변 최대 {max(max(s) for s in sizes)}, 최소 {min(max(s) for s in sizes)}, 가장 흔한 크기 {collections.Counter(sizes).most_common(3)}"]
(out/"README.md").write_text("\n".join(L)+"\n"); print("\n".join(L))
PY
