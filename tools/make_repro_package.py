"""제출·재현용 코드 패키지(zip)를 만든다. 데이터·모델·재배포 금지 파일은 넣지 않는다.

  python3 tools/make_repro_package.py      -> dist/halio_repro_YYYYMMDD.zip
"""
import fnmatch, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"installers", "_archive", "dist", "__pycache__", ".git", ".pytest_cache"}
SKIP_PATHS = ["logs/*", "jobs/logs/*", "jobs/cancelled/*", "results/preds/*.npz", "results/coco_ret/*.npz",
              "hailo_sdk*.log", "allocator.log", "results/coco_ret/RP-*", "results/preds/RP-*", "results/coco_ret/RPQ-*", "results/preds/RPQ-*", "results/*repro*",
              "*.onnx", "*.har", "*.hef", "*.npy", "*.zip",
              "98_*.bat", "99_*.bat", "jobs/running.txt", "jobs/.heartbeat"]   # 정리·삭제용 bat 과 대기소 상태 파일은 넣지 않는다
KEEP = {"logs/p1_parity.json"}          # 제출 시 P1 게이트 기록은 넣는다

def wanted(rel: str) -> bool:
    if rel in KEEP: return True
    if set(Path(rel).parts) & SKIP_DIRS: return False
    return not any(fnmatch.fnmatch(rel, p) for p in SKIP_PATHS)

out = ROOT / "dist" / f"halio_repro_{time.strftime('%Y%m%d')}.zip"
out.parent.mkdir(exist_ok=True)
n = size = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob("*")):
        if p.is_file():
            rel = p.relative_to(ROOT).as_posix()
            if wanted(rel):
                z.write(p, f"halio_project/{rel}"); n += 1; size += p.stat().st_size
print(f"{out}  파일 {n}개, 원본 {size/1e6:.1f} MB -> zip {out.stat().st_size/1e6:.1f} MB")
