import copy, re
import docx
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_COLOR_INDEX
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table

d = docx.Document("template.docx")
body = d.element.body
T = d.tables
SZ = 9  # pt, table/box text
MONO = "Consolas"

def clear_cell(cell):
    ps = cell.paragraphs
    for p in ps[1:]:
        p._p.getparent().remove(p._p)
    p = ps[0]
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    return p

def add_runs(p, text, size=SZ, mono=False, color=None):
    # **bold**, [[placeholder]] (yellow), `code`
    for tok in re.split(r"(\*\*[^*]+\*\*|\[\[[^\]]+\]\]|`[^`]+`)", text):
        if not tok:
            continue
        bold = tok.startswith("**")
        ph = tok.startswith("[[")
        code = tok.startswith("`")
        t = tok[2:-2] if (bold or ph) else (tok[1:-1] if code else tok)
        r = p.add_run(t)
        r.font.size = Pt(size if not (mono or code) else size - 0.5)
        if bold:
            r.bold = True
        if ph:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        if mono or code:
            r.font.name = MONO
            r._r.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "D2Coding")
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
    return p

def new_par_after(p, like=None):
    np_ = copy.deepcopy(like._p if like is not None else p._p)
    for r in np_.findall(qn("w:r")):
        np_.remove(r)
    for h in np_.findall(qn("w:hyperlink")):
        np_.remove(h)
    p._p.addnext(np_)
    return Paragraph(np_, p._parent)

def set_indent(p, left_cm=0.0, hanging_cm=0.0, space_after=2):
    pf = p.paragraph_format
    pf.left_indent = Cm(left_cm)
    pf.first_line_indent = Cm(-hanging_cm) if hanging_cm else None
    pf.space_after = Pt(space_after)

def fill_cell(cell, lines, size=SZ, center=False):
    """lines: list of str. '- ' bullet, '1. ' numbered, '$ ' or '    ' code, '#B ' bold heading line, '' blank."""
    p = clear_cell(cell)
    first = True
    for ln in lines:
        q = p if first else new_par_after(p, like=p)
        if not first:
            p = q
        first = False
        if ln.startswith("@IMG "):
            path, w = ln[5:].split("|")
            q.paragraph_format.alignment = 1
            q.add_run().add_picture(path, width=Cm(float(w)))
            continue
        if ln.startswith("```"):
            add_runs(q, ln[3:], size=size - 0.5, mono=True)
            set_indent(q, 0.2, 0, 0)
            q.paragraph_format.space_before = Pt(0)
            continue
        q.paragraph_format.alignment = 0
        m = re.match(r"^(\d+)\. (.*)", ln)
        if ln.startswith("- "):
            add_runs(q, "•\u00a0" + ln[2:], size)
            set_indent(q, 0.45, 0.35)
        elif ln.startswith("  - "):
            add_runs(q, "–\u00a0" + ln[4:], size)
            set_indent(q, 0.9, 0.35)
        elif m:
            add_runs(q, f"{m.group(1)}.\u00a0" + m.group(2), size)
            set_indent(q, 0.5, 0.45)
        elif ln.startswith("#B "):
            add_runs(q, "**" + ln[3:] + "**", size + 0.5)
            q.paragraph_format.space_before = Pt(4)
            q.paragraph_format.space_after = Pt(2)
        else:
            add_runs(q, ln, size)
            q.paragraph_format.space_after = Pt(3)
        if center:
            q.paragraph_format.alignment = 1

def fill_row(row, values, size=SZ):
    for c, v in zip(row.cells, values):
        if v is None:
            continue
        p = clear_cell(c)
        add_runs(p, v, size)

def add_rows(table, n):
    last = table.rows[-1]._tr
    for _ in range(n):
        last.addnext(copy.deepcopy(last))
        last = last.getnext()

def remove(el):
    el.getparent().remove(el)

# ---------------------------------------------------------------- guide text removal
for p in list(d.paragraphs):
    if p.text.startswith("▸") or p.text.startswith("자유주제는 벤치마크를 팀이 직접"):
        remove(p._p)
remove(T[0]._tbl)  # 작성 안내 box
T = d.tables
for p in d.paragraphs:
    if p.text.startswith("요약 : 어떤 모델을"):
        for r in list(p.runs):
            remove(r._r)
        r = p.add_run("요약"); r.bold = True; r.font.size = Pt(10.5)
    if p.text.startswith("7. 한계점 및 엣지"):
        p.style = d.styles["Heading 1"]
# tables now: 0 team,1 summary,2 sumbox,3 model,4 3.2,5 3.3,6 env,7 5.1,8 5.2,9 5.3,10 6.1,11 6.2,12 6.3,13 6.4,14 6.5,15 6.6,16 7.1,17 7.2,18 checklist,19 9.1,20 9.2,21 9.3,22 9.4

# ---------------------------------------------------------------- 1. 팀 정보
t = T[0]
vals = {"팀명": "[[팀명 기입]]",
        "대상 모델명": "OpenAI CLIP ViT-B/16 이미지 인코더 (open_clip `ViT-B-16`, pretrained=`openai`)",
        "대표자 연락처": "[[연락처 기입]]",
        "제출일": "2026년 [[  ]]월 [[  ]]일"}
for row in t.rows[1:]:
    k = row.cells[0].text.strip()
    if k in vals:
        fill_row(row, [None, vals[k]])
    elif k.startswith("팀원"):
        fill_row(row, [None, "[[성명 / 학번 / 역할]]"])
    elif k == "" and row.cells[1].text.strip() == "":
        fill_row(row, [None, "[[성명 / 학번 / 역할]]"])

# ---------------------------------------------------------------- 2. 결과 요약
t = T[1]
summary = [
    ["344.9", "89.8", "−74.0%"],
    ["86.19", "86.20 (분리용 행렬 포함)", "≈0% (수는 그대로, 32→8비트)"],
    ["35.13", "35.13", "0% (INT8 연산으로 수행)"],
    ["58.06 (p50)", "25.51 (p50)*", "−56.1% (2.28배)"],
    ["17.2", "39.2", "+128%"],
    ["516", "232", "−55.0%"],
    ["COCO-confirm R@1 41.80% / 최종 시험 XM3600 R@1 41.66%", "56.97% / 51.71%", "+15.17%p [+14.04, +16.32] / **+10.05%p [+9.02, +11.13]** (적응 +15.55, 양자화 −0.38)"],
]
for row, v in zip(t.rows[1:], summary):
    fill_row(row, [None] + v)
fill_cell(T[2].cell(0, 0), [
    "OpenAI CLIP ViT-B/16 이미지 인코더(사진 검색의 색인 단계)를 두 단계로 최적화했다. ① **양자화:** 원인 분석으로 찾은 **블록 5~7 fc2 입력의 이상치 32채널만 FP32 작은 행렬곱으로 분리(SPL)**하고 나머지는 정적 INT8. ② **도메인 적응:** 블록 0~7은 얼리고 뒤쪽 4블록(이미지·텍스트)만 일상 사진 21.8만 장(COCO train2017 11.8만 + CC3M 10만)으로 대조 학습(CPU 9.6시간)해, ①의 양자화 처방을 그대로 쓸 수 있게 했다.",
    "PC CPU(ONNX Runtime, 4스레드) 기준 **지연 2.28배 단축, 크기 −74%, 메모리 −55%**. 정확도는 원본 FP32 대비 **COCO 검색 R@1 +15.17%p**(적응 +15.55, 양자화 −0.38)이고, **미리 봉인해 마지막에 한 번만 연 최종 시험셋 XM3600(전 세계 36개 지역 사진)에서 +10.05%p**. Flickr30k +12.86, NoCaps +14.08%p, CIFAR-100 제로샷 68.01→68.16%.",
    "양자화 처방만 보면(원본 가중치) R@1 −0.96%p로 사전 기준(−2%p)을 통과했다. 모든 선택은 결과를 보기 전에 기록한 규칙으로 선택용 분할에서만 했고, 시험셋 사용 이력은 6.2에 모두 공개한다. Hailo-10H(DFC 5.4.0)에는 원본 가중치로 이식해 공식 설정과 같은 13 컨텍스트로 컴파일했으나 판정 분할 R@1 −3.03%p(공식 −3.16%p)로 기준 미달이며, 적응 가중치 이식은 본선에서 한다(6.6·7장).",
    "재현: 원 실험 PC의 빈 폴더와 다른 PC(새 환경)에서 정확도 값이 모두 그대로 재현됐다(6.3). *최종 구성은 SPL과 연산 그래프가 같아(가중치 값만 다름) 속도·메모리는 SPL 측정값이다(적응 모델 직접 측정도 2.30배).",
])

# ---------------------------------------------------------------- 3.1
t = T[3]
m31 = {"모델명 / 버전": "CLIP ViT-B/16 이미지 인코더 (open_clip `ViT-B-16`, QuickGELU, 입력 224×224, 출력 512차원)",
       "모델 유형 (비전 / 언어 / 멀티모달 등)": "멀티모달(이미지-텍스트 대조 학습) 모델의 이미지 인코더 = 비전 트랜스포머(12블록, 폭 768)",
       "저장소 URL": "https://github.com/mlfoundations/open_clip (원 모델: https://github.com/openai/CLIP)",
       "커밋 해시 또는 릴리스 태그": "open_clip_torch 3.3.0 (PyPI 릴리스) / Hailo 경로: Hailo Model Zoo 5.4.0 공식 ONNX",
       "사전학습 가중치 출처": "OpenAI 공개 CLIP 가중치 (open_clip `pretrained=openai`로 자동 다운로드) = 기준 모델. 최종 모델은 여기에 이미지·텍스트 인코더 뒤쪽 4블록만 도메인 적응 학습한 가중치(`repro/weights/da5_chosen.pt`, 5.3)를 덮어쓴 것",
       "라이선스": "OpenAI CLIP: MIT, open_clip: MIT [[제출 전 각 저장소 LICENSE 확인]]"}
for row in t.rows[1:]:
    k = row.cells[0].text.strip()
    if k in m31:
        fill_row(row, [None, m31[k]])

# ---------------------------------------------------------------- 3.2
fill_cell(T[4].cell(0, 0), [
    "**한 줄 요약:** 실제 사진 검색 앱에서 쓰이는 공개 모델이고, Hailo-10H 공식 지원 모델이라 제조사 기본 양자화와 같은 평가로 직접 비교할 수 있으며, 그 기본 양자화에서도 손실이 큰 모델이라 개선 여지가 분명하다.",
    "#B 왜 이 모델인가",
    "- **실사용:** 오픈소스 사진 관리 앱 Immich의 문장 검색이 CLIP 계열 모델을 쓰며, 선택 가능한 모델에 `ViT-B-16__openai`가 있다 [1].",
    "- **Hailo 공식 지원 = 비교 기준 확보:** Hailo Model Zoo의 Hailo-10H 공개 모델 표에 `clip_vit_b_16_image_encoder`가 있다(CIFAR-100 제로샷 top-1 FP 68.6 → 하드웨어 64.6) [2]. 같은 평가에서 제조사 기본 설정 대비 개선을 보일 수 있다.",
    "- **개선 여지:** 같은 표에서 ViT-B/16의 양자화 손실(−4.0%p)은 ViT-B/32(−0.8%p)보다 크다. B/16은 FP 정확도가 더 높다(68.6 vs 65.3) [2].",
    "- **기술적 난이도:** 트랜스포머의 활성값 이상치는 8비트 양자화를 어렵게 하는 알려진 문제다 [3][6]. 실제로 ONNX Runtime 기본 정적 INT8에서 R@1이 0.19%로 붕괴했다(5.3).",
    "#B 왜 경량화해야 하나",
    "- **비용은 색인에 몰린다:** 사진 검색은 사진마다 이미지 인코더를 한 번(색인), 질의마다 텍스트 인코더를 한 번 돌린다. 실측에서 질의 1개는 27 ms(p50), 사진 10만 장과의 비교 계산은 4 ms인 반면, 색인은 FP32에서 사진 1장당 71.6 ms였다(6.5). 갤러리가 클수록 이미지 인코더가 비용을 지배한다.",
    "- **실시간 처리 가능 여부:** 디코드를 포함한 사진 1장 처리가 FP32 71.6 ms에서 INT8 31.5 ms로 줄어 30 fps(33.3 ms) 안에 들어온다(여유는 작다). 1,000장 색인 최대 메모리는 508→213 MB.",
    "- **기기 안 처리:** 사진을 서버로 보내지 않아 개인정보를 보호하고 오프라인에서도 동작한다. NPU로 옮기면 CPU를 다른 작업에 쓸 수 있다.",
    "- **같은 구조를 유지:** 모델 구조·크기·연산은 원본 그대로이고 가중치 값만 바뀐다. 그래서 Hailo 공식 지원 모델의 컴파일 경로와 원본으로 만든 양자화 처방을 그대로 쓴다. 텍스트 인코더도 뒤쪽 4블록을 함께 적응했으므로(FP32, 호스트 CPU), 앱에서는 이미지·텍스트 인코더를 한 쌍으로 바꾸고 색인을 한 번 다시 만든다(6.5의 R@1이 이 조합의 측정값). 더 작은 다른 모델(예: MobileCLIP [5])로 바꾸는 것과 달리 Hailo 이식·양자화 분석을 다시 할 필요가 없다.",
])

# ---------------------------------------------------------------- 3.3
fill_cell(T[5].cell(0, 0), [
    "**결론:** CPU 폴백 없이 그래프 전체가 Hailo-10H용 HEF로 컴파일됐다(DFC 5.4.0, 에뮬레이터로 정확도 검증). 실기기 동작·속도는 본선에서 확인한다. 또한 예선 단계의 기법 선택 자체를 NPU 구동을 전제로 내렸다(아래 마지막 항목).",
    "#B 컴파일 가능성 확인",
    "- **파싱:** Hailo Model Zoo 공식 ONNX를 시작·끝 노드 지정 없이 `translate_onnx_model`로 변환했다. 앞뒤를 잘라 CPU에 남긴 부분이 없다.",
    "- **구조 확인(컴파일된 HAR):** 입력 224×224×3(uint8) → 출력 512차원. 입력 정규화가 칩 안에 있고 호스트·후처리 층이 없다(공식·우리 구성 모두). 층 구성은 conv 74, matmul 24, 정규화·활성·원소 연산 등 표준 연산뿐이다(`logs/dfc/hef_check.md`).",
    "- **컴파일 결과:** 공식 설정 HEF 79.3 MB·13 컨텍스트, 우리 구성(이상치 갈래 16비트) 79.6 MB·13 컨텍스트로 공식과 같다. 참고로 7번 층 전체를 16비트로 두면 14 컨텍스트가 된다.",
    "- **정밀도 대응:** Hailo-10H는 INT8과 16비트(a16_w16)를 층별로 섞을 수 있어, ORT에서 FP32로 남긴 이상치 채널 갈래를 16비트로 대응했다. 활성 16·가중치 8비트(a16_w8)는 DFC 내부 오류로 컴파일되지 않아 쓰지 않았다. INT4(가중치 전용)는 ORT에서 시험했으나 정확도가 회복되지 않아 제외했다(5.3).",
    "- **텍스트 인코더:** 질의마다 한 번이라 호스트 CPU(FP32, 적응 가중치)에 둔다. HEF 컴파일도 시도했으나 공식 설정의 미세조정 단계가 이 PC에서 멈춰 중단했다(7.1).",
    "- **본선 기준값:** Hailo가 공식 HEF로 잰 값은 배치 1 51.5 fps, 배치 8 158 fps다 [2]. 우리 구성은 컨텍스트 수가 같아 비슷한 속도를 기대하되, 보드에서 실측해 확정한다. (DFC 프로파일러의 컨텍스트별 추정을 합산한 값은 전환 비용이 빠져 과대 추정이므로 쓰지 않는다.)",
    "#B 실기기 구동을 위해 내린 설계 선택",
    "- **NPU에서 돌 수 있는 방식만 채택:** CPU에서는 동적 INT8+fc2 보호(A1-c67-pc)가 더 정확하고 빨랐지만, 실행 중 스케일을 계산하는 방식이라 NPU에 올릴 수 없어 최종 후보에서 뺐다. 채택 처방(SPL)은 고정 스케일 정적 INT8이다(5.3, 7.1).",
    "- **연산 전부를 칩에:** 이상치 처리를 CPU로 빼지 않고, 등가 변환(채널 재배열+갈래 분할)으로 그래프 안에 넣어 칩 안에서 처리한다. 입력 정규화도 칩 안이라 호스트는 사진 디코드·리사이즈만 한다.",
    "- **속도 비용 최소화:** 7번 층 전체를 16비트로 두면 14 컨텍스트로 늘어 컨텍스트 전환이 추가된다. 32채널 갈래만 16비트로 두는 분할을 택해 공식과 같은 13 컨텍스트를 유지했다.",
    "- **칩의 혼합 정밀도 활용:** 대부분은 INT8, 이상치 갈래만 16비트로 두어 Hailo-10H의 층별 정밀도 지정 기능을 필요한 곳에만 썼다.",
    "- **메모리 여유:** HEF 79.6 MB로 온보드 8 GB 메모리에 비해 작아, 색인용 임베딩과 다른 작업을 함께 둘 여유가 있다.",
    "- **역할 분담:** 사진마다 도는 이미지 인코더(비용 대부분, 3.2)는 NPU, 질의당 1회인 텍스트 인코더는 호스트 CPU에 둔다. 색인은 사진을 몰아서 처리하는 작업이라 배치·비동기 처리로 NPU를 쉬지 않게 할 수 있다(배치 8 공개값 158 fps [2]). 본선에서 배치·파이프라인 효과와 CPU 단독 대비 CPU 부하를 실측한다(7.2).",
])

# ---------------------------------------------------------------- 4. env
t = T[6]
env = {"CPU": "Intel Core i9-14900KF (측정은 4스레드 고정) / 재현 확인: Intel Core i9-12900",
       "GPU (모델 / VRAM)": "사용하지 않음 (정확도·속도·Hailo 컴파일 모두 CPU)",
       "RAM": "23.5 GB (WSL2 할당)",
       "OS": "Windows + WSL2 Ubuntu 22.04 (커널 6.18.33.2-microsoft-standard-WSL2)",
       "Python 버전": "3.10.12",
       "딥러닝 프레임워크 / 버전": "PyTorch 2.14.1+cpu, open_clip_torch 3.3.0, ONNX 1.23.1 (opset 17), NumPy 2.2.6, Pillow 12.3.0",
       "CUDA / cuDNN 버전": "해당 없음 (CPU 전용, `CUDA_VISIBLE_DEVICES=''`)",
       "양자화·최적화 도구 / 버전": "ONNX Runtime 1.23.2 (onnxruntime.quantization, QDQ) / Hailo Dataflow Compiler 5.4.0 (TensorFlow 2.19.1, hailo10h)"}
for row in t.rows[1:]:
    k = row.cells[0].text.strip()
    if k in env:
        fill_row(row, [None, env[k]])

# ---------------------------------------------------------------- 5.1
t = T[7]
add_rows(t, 3)
r51 = [
    ["ONNX 변환·상수 폴딩", "이미지 인코더 전체", "opset 17, 노드 996→489", "출력 일치 (최대 오차 1.03e-5, 코사인 1.000000)"],
    ["정적 INT8 PTQ (가중치 MatMul 한정, A2m)", "가중치 MatMul 49개", "MinMax, 텐서별, QDQ, 캘리브 COCO 64장", "R@1 −6.89%p (ORT 기본 정적은 −41.61%p 붕괴)"],
    ["혼합 정밀도 (민감층 FP32 유지)", "블록 6·7 fc2", "해당 층만 양자화 제외", "R@1 −6.89 → −1.18%p"],
    ["채널별 가중치 양자화", "가중치 MatMul", "per-channel", "R@1 −1.18 → −0.86%p"],
    ["이상치 채널 분리 (SPL-b567-k32, 채택 처방)", "블록 5·6·7 fc2 입력", "캘리브에서 |x| 상위 32채널만 FP32 작은 행렬곱, 나머지 INT8", "R@1 −0.96%p, 25.51 ms (2.28배), 89.8 MB (fc2 통째 FP32보다 14 MB 작음)"],
    ["**도메인 적응 + SPL (최종, DA5-SPL)**", "이미지 블록 8~11·ln_post·proj + 텍스트 블록 8~11·ln_final·투영 학습 → 위 SPL 처방", "COCO train2017 전체 11.8만 + CC3M 10만 장, 대조 학습 2에폭, lr 1e-5, 블록 0~7 고정, α=1.0", "적응 FP32 +15.55%p, 양자화 −0.38%p → 원본 대비 +15.17%p (최종 시험 XM3600 +10.05)"],
    ["(Hailo) 이상치 갈래 16비트", "블록 7 fc1·GELU·fc2의 32채널", "a16_w16, 나머지 공식 설정", "13 컨텍스트 유지, R@1 −3.03%p (공식 −3.16)"],
]
for row, v in zip(t.rows[1:], r51):
    fill_row(row, v)

# ---------------------------------------------------------------- 5.2
t = T[8]
q52 = {"양자화 방식 (PTQ / QAT)": "PTQ (정적). 양자화 전에 FP32 상태에서 도메인 적응 학습(뒤쪽 4블록)을 하고, 그 FP32 모델을 원본과 같은 처방으로 양자화. 뒤쪽 4블록 QAT(fake-quant 증류)도 시험했으나 최종 적응 모델에서 이득이 없어(−0.00%p [−0.51, +0.48]) 채택하지 않았다(5.3)",
       "정밀도 (INT8 / INT4 / 혼합)": "혼합: 가중치 INT8(채널별)·활성 INT8(텐서별, MinMax) + 이상치 32채널×3블록 FP32 / Hailo: INT8 + 이상치 갈래 a16_w16",
       "캘리브레이션 데이터셋": "COCO val2017 중 평가 분할에서 제외한 사진 (`splits/calib_coco_64_s0.txt`). 평가 사진과의 겹침을 코드로 차단(겹침 0)",
       "캘리브레이션 샘플 수": "64장 (Hailo 사후 분석: 64장 3묶음, 합집합 189장)",
       "양자화 제외 레이어 (있는 경우)": "비가중치 연산(어텐션 활성 MatMul 24개, Softmax, LayerNorm, Add)은 FP32 / 이상치 채널 행렬곱 3개(블록 5·6·7) FP32. 적응 후에도 보호 노드 목록·이상치 채널이 원본과 동일(블록 0~7을 얼렸기 때문)"}
for row in t.rows[1:]:
    k = row.cells[0].text.strip()
    if k in q52:
        fill_row(row, [None, q52[k]])

# ---------------------------------------------------------------- 5.3
fill_cell(T[9].cell(0, 0), [
    "원인을 측정으로 찾고, 찾은 곳만 고치는 순서로 진행했다. 정확도 수치는 CIFAR-100 검증셋(10,000장) 기준이다(검색 평가는 6장).",
    "1. **첫 실패:** ORT 기본 정적 INT8(A2)이 2.09%로 붕괴했다. 채널별·Percentile·Entropy·일부 층 제외 등 7가지 탐침이 모두 1.5~4.8%에 그쳐, 범위 추정 방식이 원인이 아님을 확인했다. Softmax·Add까지 양자화한 것이 원인으로 보고 가중치 MatMul 49개로 제한하자(A2m) 53.6%로 회복했다(FP32 68.8%).",
    "2. **어느 층인가(민감도):** 층 종류·깊이·블록별로 한 묶음씩 FP32로 남겨 회복량을 쟀다. fc2 12개를 남기면 손실의 100%, 블록별로는 7번 75%·6번 49%·5번 15%가 회복됐다.",
    "3. **왜 그 층인가(활성값):** 블록 5~7 fc2 입력은 최대값이 17/26/42인데 99.9%의 값이 8비트 255칸 중 9~18칸에 몰려 있다(그림 2). 이상치는 3~4개 채널에 집중돼 있다.",
    "4. **처방 1, 민감층 보호:** 6·7번 fc2를 FP32로 두고 가중치를 채널별로 양자화하자 69.09%로 FP32 수준이 됐다.",
    "5. **처방 2, 이상치 채널 분리(최종):** 층 전체가 아니라 이상치 32채널만 FP32 행렬곱으로 떼어 내고 나머지를 INT8로 두었다. 원본과 등가 변환인지 확인했다(코사인 0.9999999). FP32로 남는 부분이 작아 모델이 더 작다(89.8 MB vs 103.7 MB).",
    "6. **Hailo 이식:** fc2만 16비트로 나누면 실패했다(fc1+GELU 출력이 먼저 8비트로 깎임). 그래서 fc1부터 갈래를 나눠 13 컨텍스트를 유지하며 select 분할에서 기준을 통과했다. 판정용 분할 결과는 6.6에서 다룬다.",
    "7. **선택 규칙:** 후보는 NPU에서 실행 가능한 정적 구성으로 한정했다. 동적 INT8+fc2 보호(A1-c67-pc)는 CPU에서 더 정확하고 빠르지만(−0.57%p, 24.05 ms) 실행 중 범위를 계산하는 방식이라 NPU 이식 전제에 맞지 않아 비교 기준으로만 남겼다.",
    "8. **도메인 적응(10/20~10/24):** 범용 CLIP을 일상 사진 검색에 맞게 뒤쪽만 추가 학습했다. 설계 근거는 셋이다. ① **양자화 처방 보존:** 이상치가 블록 5~7에 있으므로 블록 0~7을 얼리면 SPL 처방이 그대로 유효하다(적응 후 보호 노드·이상치 채널 동일, 블록 8~11에도 새 이상치 없음). ② **CPU로 가능:** 블록 7 출력을 한 번 저장해 두고 뒤쪽 4블록만 학습해 GPU 없이 PC CPU로 끝냈다. ③ **일반성 유지:** CLIP 사전학습과 같은 대조 손실로 학습하고 [9], 원본과 학습본을 비율 α로 섞어(WiSE-FT [8]) CIFAR 하락이 1%p 이내인 α만 고른다.",
    "9. **단계별 확장(모두 사전 등록한 규칙으로 채택 판정):** COCO 3만 장 이미지 쪽만(원본 대비 +13.1%p, FP32·select) → 텍스트 인코더 뒤쪽도 함께(+1.5%p) → COCO 전체 11.8만 장(+2.3%p [+1.7, +3.0]) → COCO 밖 웹 사진 CC3M 10만 장 추가(COCO select 유지 −0.11, **COCO 밖 검증셋 CC3M val +4.30%p [+2.70, +5.85]**). 마지막 단계는 COCO select로는 COCO 밖 일반화를 잴 수 없어, 선택용 검증셋(CC3M val 2,000장)을 따로 정하고 규칙을 먼저 기록했다.",
    "   **과적합·평가 맞춤 방지:** 학습은 COCO train2017·CC3M train만 쓰고 평가 사진과 분리했다. 각 단계마다 설정·α 선택 규칙·채택 조건(현 후보 대비 차이의 95% 하한 > 0 등)을 실행 전에 `docs/02`에 기록했다. 최종 구성은 동결 후 미리 봉인해 둔 XM3600 [12]으로 단 한 번 판정했다(6.2).",
    "#B 기각한 시도 (근거 포함)",
    "- **INT4 가중치 전용:** CPU 속도 이득이 없고(60 ms), fc2를 보호해도 정확도가 회복되지 않았다(가중치 오차라는 다른 실패 양상).",
    "- **QOperator 형식:** QDQ와 속도 차이가 없어 QDQ를 유지했다.",
    "- **CIFAR 사진으로 캘리브:** CIFAR 정확도는 올랐지만 실제 사진 검색 R@1이 23.7%로 무너졌다. 같은 도메인 효과였고, 이 결과로 평가 축을 검색으로 바꿨다(6.2).",
    "- **테스트 시 레지스터[4]:** 원인 설명(이상치 = 레지스터 뉴런)은 확인했으나, 정확도 유지와 범위 축소를 동시에 만족하는 설정이 없었다(−2.5~−16%p).",
    "- **블록 제거(ReplaceMe):** 블록 1개 제거도 R@1 −1.26%p [−2.22]로 기준 미달.",
    "- **입력 해상도 축소:** 192에서 −1.14%p로 경계였다. 적응 모델에서도 192+SPL이 −1.50%p [−2.21]로 기준 미달이라 제외했다.",
    "- **뒤쪽 4블록 QAT(fake-quant 증류):** 원본 가중치에서는 select +1.50%p였지만 FP32로 같은 학습을 해도 +1.48%p라, 이득은 양자화 손실 회복이 아니라 추가 조정 효과였다. 최종 적응 모델에서는 이득이 없었다(−0.00%p).",
    "- **이미지·텍스트 α 따로 + 체크포인트 평균:** 50개 조합 중 최선이 +0.49%p [−0.02, +0.98]로 채택 조건(하한 > 0) 미달. 평균(soup)은 모든 조합에서 이득 없음.",
    "- **Hailo 캘리브를 train2017 256장으로:** 오히려 나빠졌다(−1.86 vs −0.48%p, select).",
])

# ---------------------------------------------------------------- 6.1
t = T[10]
b61 = {"데이터셋 / 벤치마크명": "(주) COCO val2017 텍스트→이미지 검색, 자체 분할 / **(최종 시험) XM3600 영어 캡션 [12]** / (COCO 밖 확인) Flickr30k 1K test [10], NoCaps val [11] / (보조) CIFAR-100 제로샷 / (선택용) CC3M val 2,000장 [13]",
       "공개 여부 및 출처 URL": "모두 공개. COCO: https://cocodataset.org (평가 val2017, 적응 학습 train2017) / XM3600: https://google.github.io/crossmodal-3600 / Flickr30k 1K: huggingface.co/datasets/clip-benchmark/wds_flickr30k / NoCaps val: huggingface.co/datasets/lmms-lab/NoCaps / CC3M: huggingface.co/datasets/pixparse/cc3m-wds (적응 학습 10만 장 + 선택용 val) / CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html",
       "평가셋 규모 (샘플 수)": "COCO val2017 5,000장 = 캘리브 제외 189 / 선택용 select 2,405장(캡션 12,035) / 판정용 confirm 2,406장(캡션 12,033). XM3600: 3,600장(36개 지역×100, 영어 캡션 7,200), NoCaps와 겹치는 43장을 뺀 3,557장이 주 지표. Flickr30k 1,000장(캡션 5,000), NoCaps 4,500장(캡션 45,000), CC3M val 2,000장. CIFAR-100: 검증 10,000 / test 10,000",
       "평가 지표": "Recall@1(주), R@5, R@10 = 캡션으로 검색했을 때 정답 사진이 상위 K장 안에 있는 비율(이미지별 평균). 차이는 이미지 단위 짝지은 bootstrap 2,000회 95% 범위. 보조: CIFAR top-1",
       "지표 선정 사유": "목표 응용(문장으로 사진 찾기)을 그대로 측정한다. 텍스트 인코더는 FP32로 두어(양자화하지 않음) 이미지 인코더 양자화의 영향만 분리한다. XM3600은 COCO와 출처·지역이 다른 전 세계 사진이라 COCO 맞춤 효과를 걸러 내는 최종 시험으로 골랐다"}
for row in t.rows[1:]:
    k = row.cells[0].text.strip()
    if k in b61:
        fill_row(row, [None, b61[k]])

# ---------------------------------------------------------------- 6.2
fill_cell(T[11].cell(0, 0), [
    "- **응용을 직접 잰다:** R@K는 \"검색 결과 상위 K장 안에 원하는 사진이 있나\"다. R@1을 주지표로, 화면 첫 줄에 해당하는 R@5·R@10을 함께 보고한다.",
    "- **결과를 보기 전에 규칙을 고정했다:**",
    "  - 분할(제외 189 / select / confirm)을 첫 평가 전에 파일로 고정하고 sha256을 기록했다. 캘리브 사진은 평가 사진과 겹치지 않게 코드로 막았다.",
    "  - 구성은 선택용 분할(COCO select·CIFAR val, 마지막 단계는 CC3M val 추가)로만 골랐다. 시험셋은 `--i-know-final` 없이는 열리지 않게 코드로 잠갔다.",
    "  - 차이는 같은 이미지끼리 짝지은 bootstrap 95% 범위로 보고한다. \"FP32와 거의 같다\"의 기준(하한 ≥ −2%p)은 갤러리 약 2,400장의 측정 오차(약 ±1.4%p)를 고려해 정했다.",
    "  - 우리 ONNX·텍스트 임베딩이 Hugging Face 공식 CLIP과 같은지 확인했다(코사인 1.000000).",
    "- **시험셋 사용 이력(전부 공개):** ① 10/17 양자화 처방 동결 → confirm·CIFAR test 1회. ② 10/20 적응 추가 → 새 구성 판정에 confirm을 한 번 더 사용. ③ 10/21 중간 동결(DA2) → confirm·Flickr30k·NoCaps를 열었다(NoCaps는 팀 확인 전에 열린 운영 실수). ④ 이후 추가 학습은 선택용 분할만으로 고르고, 새 시험셋 XM3600을 데이터만 확인한 채 봉인하며 판정 규칙(원본 대비 하한 ≥ 0이면 제출, < −2이면 이전 구성으로 복귀 등)을 먼저 기록했다. ⑤ 10/25 최종 동결 후 XM3600을 한 번 열었다. **따라서 최종 구성에 대한 깨끗한 시험은 XM3600뿐이고**, confirm·Flickr30k·NoCaps의 최종 구성 수치는 이미 열린 셋에 대한 참고값이다.",
    "- **분류만으로는 공정하지 않았다:** 초기에는 CIFAR-100으로 실험했다. 그런데 CIFAR 사진으로 캘리브한 Hailo 모델이 CIFAR 정확도는 오르면서 실제 사진 검색 R@1은 23.7%로 무너졌다(FP 40.9%). 그래서 검색을 주지표로 바꾸고 CIFAR는 보조로 남겼다.",
    "- **추가 학습이 평가를 오염시키지 않게:** 적응 학습은 COCO train2017·CC3M train만 쓰고 평가 사진과 겹치지 않는다. COCO 계열 상승폭은 과대 추정될 수 있어, 출처가 다른 XM3600(Open Images, 36개 지역)으로 최종 판정했다.",
    "#B 선택하지 않은 대안",
    "- **ImageNet 제로샷 분류:** 분류라 응용(검색)과 다르고 CIFAR와 같은 한계가 있다.",
    "- **공개 검색 분할(COCO 5K 등):** 캘리브·선택·판정 사진을 분리하기 어려워 자체 분할을 썼다. 그래서 공개 논문 수치와 직접 비교하지 않는다.",
    "- **속도만 보기:** 정확도 조건 없이 빠른 구성(예: 동적 INT8 −1.48%p)은 마진 하한(−2.18)을 넘지 못해 탈락시켰다.",
])

# ---------------------------------------------------------------- 6.3
fill_cell(T[12].cell(0, 0), [
    "- **데이터:** 모두 공개 데이터이고 공식 URL에서 자동으로 받는다(약 1.2 GB). 비공개 데이터는 없다.",
    "- **명령 하나로 재현:** `bash reproduce.sh` = 다운로드 → 분할 재생성과 제출 분할 sha256 비교 → FP32 export와 PyTorch 출력 일치 검사 → 양자화 → 적응 가중치 sha256 확인 후 최종 구성 생성 → COCO-confirm·CIFAR test(5구성) → 속도 → 기대값과 비교 표 출력(9장).",
    "- **적응 가중치:** 학습 결과(`repro/weights/da5_chosen.pt`, 이미지·텍스트 뒤쪽 4블록, 166 MB, sha256 기록)를 GitHub Release로 배포한다. 직접 학습해 보려면 `DA_TRAIN=1`(CPU 약 15시간, COCO train2017·CC3M 다운로드, 디스크 약 90 GB). 학습 코드·설정·로그도 함께 제출한다.",
    "- **시험셋 평가:** XM3600·NoCaps·Flickr30k는 `src/xm3600_eval.py`, `src/nocaps_eval.py`, `src/flickr_eval.py`로 같은 방식으로 다시 잴 수 있다(공식 주소·Hugging Face 공개 사본에서 자동 다운로드).",
    "- **소규모 서브셋(양식 6.3 요구):** `bash repro_quick.sh`(약 20~30분, 다운로드 약 0.3 GB) = confirm 앞 500장·CIFAR test 앞 2,000장으로 4구성을 처음부터 다시 만들어 기대값과 비교. 기대값은 전체 실행의 저장 결과에서 같은 부분을 잘라 계산한 값과 소수 넷째 자리까지 같다.",
    "- **검증 결과:** 양자화 처방은 원 실험 PC 빈 폴더 2회와 **다른 PC(i9-12900, 새 WSL·새 파이썬 환경)** 1회에서 정확도 8개 값이 모두 같았다(속도 배율 2.28/2.37배). 최종 구성(DA5-SPL) 반영 후(10/25) 원 실험 PC의 **빈 저장 폴더에서 데이터 다운로드부터 다시 실행해 4구성 16개 값이 모두 일치**했고, 빠른 재현 16개 값도 일치했다(전체 실행 임베딩의 앞 500장으로 계산한 값과도 소수 넷째 자리까지 같음).",
    "- **제출물:** 코드는 GitHub 저장소, 적응 가중치·모델 파일(ONNX·HEF)은 Release·공유 링크(8장), 재현 가이드(`docs/06_재현가이드.md`). GPU 불필요, 전체 2~4시간.",
])

# ---------------------------------------------------------------- 6.4
t = T[13]
p64 = {"입력 해상도": "224×224 (CLIP 표준 전처리: 리사이즈·중앙 크롭·정규화)",
       "배치 크기": "1",
       "Warm-up 횟수": "20회",
       "측정 반복 횟수": "300회 × 3반복 (반복마다 구성 순서를 섞음)",
       "통계 방식 (평균 / p50 / p95)": "p50(대표값), p95 병기, 3반복의 중앙값. 메모리 = 측정 프로세스 최대 상주 메모리(VmHWM)",
       "측정 도구 및 방법": "ONNX Runtime 1.23.2 CPUExecutionProvider, intra-op 4스레드·inter-op 1·순차 실행, 구성마다 새 프로세스, 부하가 낮을 때 측정. 색인 실측은 사진 1,000장 디코드+전처리+인코더(3반복 중앙값)"}
for row in t.rows[1:]:
    k = row.cells[0].text.strip()
    if k in p64:
        fill_row(row, [None, p64[k]])

# ---------------------------------------------------------------- 6.5
t = T[14]
add_rows(t, 4)
r65 = [
    ["정확도 지표 1 (COCO-confirm R@1)", "41.80%", "56.97%", "+15.17%p [+14.04, +16.32]"],
    ["정확도 지표 2 (CIFAR-100 test top-1)", "68.01%", "68.16%", "+0.15%p"],
    ["추론 Latency (ms, p50 / p95)*", "58.06 / 60.02", "25.51 / 26.52", "−56.1% (2.28배)"],
    ["Throughput (FPS, 인코더 단독)", "17.2", "39.2", "+128%"],
    ["모델 크기 (MB)", "344.9", "89.8", "−74.0%"],
    ["피크 메모리 (MB)", "516", "232", "−55.0%"],
    ["**정확도 지표 3 (최종 시험 XM3600 R@1, 3,557장)**", "41.66%", "51.71%", "**+10.05%p [+9.02, +11.13]**"],
    ["정확도 지표 4 (COCO-confirm R@5 / R@10)", "68.10% / 78.54%", "82.48% / 90.00%", "+14.39 / +11.46%p"],
    ["정확도 지표 5 (Flickr30k 1K / NoCaps val R@1)", "61.52% / 46.87%", "74.38% / 60.95%", "+12.86 / +14.08%p"],
    ["사진 1,000장 색인 (디코드 포함)*", "71.6 s · 14.0장/s · 508 MB", "31.5 s · 31.7장/s · 213 MB", "2.27배 · −58%"],
]
for row, v in zip(t.rows[1:], r65):
    fill_row(row, v)

# Hailo table: copy of 6.5 table structure
ht = copy.deepcopy(t._tbl)
hailo = Table(ht, t._parent)
for _ in range(len(hailo.rows) - 7):
    remove(hailo.rows[-1]._tr)
fill_row(hailo.rows[0], ["항목 (Hailo-10H, DFC 에뮬레이터)", "Hailo FP", "공식 설정", "우리 처방 (H-splf-b7-k32)"])
hrows = [["COCO-confirm R@1 (FP 대비)", "41.80%", "38.65% (−3.16%p)", "38.77% (−3.03%p)"],
         ["COCO-confirm R@5 / R@10", "68.10 / 78.54%", "64.85 / 76.26%", "65.89 / 77.09%"],
         ["CIFAR-100 test top-1", "68.01%", "61.96%", "65.58%"],
         ["캘리브 묶음 3개 confirm R@1 평균 [범위] (사후)", "—", "−3.16 (1묶음)", "−2.26 [−3.03, −1.41]"],
         ["컨텍스트 수 / HEF 크기", "—", "13 / 79.3 MB", "13 / 79.6 MB"],
         ["속도", "—", "Hailo 발표 51.5 fps (배치 1)", "본선 보드 실측 예정"]]
for row, v in zip(hailo.rows[1:], hrows):
    fill_row(row, v)

# figure + caption + hailo table after 6.5 table
anchor = t._tbl
cap_like = None
for p in d.paragraphs:
    if p.text == "" and p._p.getprevious() is anchor:
        cap_like = p
        break
p_fig = new_par_after(cap_like)
p_fig.paragraph_format.alignment = 1
p_fig.add_run().add_picture("fig1_pareto.png", width=Cm(15.5))
p_cap = new_par_after(p_fig)
add_runs(p_cap, "* 최종 구성(DA5-SPL)은 SPL과 연산 그래프·크기가 같아(489노드, INT8 MatMul 49개, 89.8 MB) 속도·메모리는 SPL 실측값(적응 모델 직접 측정 2.30배). XM3600 외 시험셋은 최종 구성 이전에 열린 적이 있다(6.2). 그림 1은 양자화 처방 비교(원본 가중치, 같은 세션 측정, 95% 범위). 빈 원은 NPU 이식 전제에 맞지 않는 동적 INT8(비교 기준).", 8.5, color="555555")
p_cap.paragraph_format.alignment = 1
p_h = new_par_after(p_cap)
add_runs(p_h, "**Hailo-10H 이식 결과 — 원본 가중치 기준** (같은 COCO-confirm 분할, Hailo FP = 양자화 전 Hailo 그래프. 적응 가중치 이식은 본선)", 9.5)
p_h.add_run()
p_h._p.addnext(ht)
p_hn = Paragraph(copy.deepcopy(p_h._p), p_h._parent)
for r in p_hn._p.findall(qn("w:r")):
    p_hn._p.remove(r)
ht.addnext(p_hn._p)
add_runs(p_hn, "Hailo 정확도는 DFC 에뮬레이터 값이다. 속도는 보드가 없어 측정하지 않았고, 공식 HEF의 속도는 Hailo 발표값이다 [2].", 8.5, color="555555")

# ---------------------------------------------------------------- 6.6
fill_cell(T[15].cell(0, 0), [
    "#B 정확도 변화의 분해: 적응 +15.55, 양자화 −0.38 (COCO-confirm)",
    "@IMG fig4_da.png|14",
    "- **적응 효과는 평가셋 전반에서 일관된다(적응 FP32, 원본 대비):** COCO +15.55, Flickr30k +13.84, NoCaps +14.88, **최종 시험 XM3600 +10.78%p**. CIFAR(32×32 저해상도 분류)는 67.69%(−0.32%p)로 범용성도 크게 잃지 않았다.",
    "- **양자화 손실은 적응 전과 같은 수준이다:** COCO −0.38 [−0.86, +0.11], Flickr −0.98, NoCaps −0.79, XM3600 −0.74%p(적응 전 SPL은 −0.96~−1.58). 블록 0~7을 얼려 이상치 위치가 그대로이고(보호 노드 동일), 학습한 블록 8~11에도 새 이상치가 생기지 않았다. 원인 분석으로 만든 처방이 적응 후에도 그대로 통한다.",
    "- **숫자의 의미:** XM3600 +10.05%p는 전 세계 사진 검색에서 질의 100개 중 약 10개가 더 1등으로 찾힌다는 뜻이다. R@10은 76.69→85.34%로, 열 장 안에 정답이 없던 질의의 약 37%가 해결됐다.",
    "- **COCO 맞춤 몫:** 같은 최종 모델이 COCO에서 +15.2, XM3600에서 +10.1%p다. 차이(약 5%p)는 COCO식 사진·캡션에 맞춰진 몫으로 본다.",
    "#B 추가 개선 라운드는 최종 시험에서 구분되지 않았다 (정직한 결과)",
    "- 중간 동결(DA2, COCO 3만 장) 이후 데이터 확대(DA3)와 CC3M 추가(DA5)는 선택용 분할에서 각각 select +2.32, CC3M val +4.30%p로 조건을 통과했다. 그러나 XM3600에서는 DA2-SPL 51.43 / DA3-SPL 52.14 / DA5-SPL 51.71%로 차이가 오차 범위 안이다(DA5 vs DA2 +0.28 [−0.58, +1.14]).",
    "- 즉 추가 이득은 선택용 검증셋과 분포가 가까운 쪽(COCO, 웹 사진)에서만 나타났다. 원본 대비 +10%p라는 큰 효과는 확실하지만, 그 위의 미세 개선은 검증셋 분포에 맞춘 부분이 있다. 미리 봉인한 시험셋이 있었기에 이를 구분할 수 있었다.",
    "#B 손실은 어디서 나는가 (양자화)",
    "- 손실은 모델 전체가 아니라 **블록 5~7 fc2 입력의 소수 이상치 채널**에서 생긴다(5.3의 민감도, 그림 2). 이상치는 사진당 약 5개 토큰에서만 켜지는 MLP 뉴런이 만들고, 활성이 가장 큰 뉴런 상위 10개가 모두 우리가 분리한 32채널 안에 있었다. 이는 비전 트랜스포머의 '레지스터 뉴런' 보고 [4]와 같은 현상이며, 비전 인코더의 8비트 민감층이 중간 블록 FC2에 몰린다는 보고 [6]와도 일치한다.",
    "@IMG fig2_mechanism.png|13.5",
    "- **단계별 기여(그림 3, 사후 평가):** 보호 없는 정적 INT8의 손실 −6.89%p 중 대부분이 6·7번 fc2 보호 한 단계에서 회복된다(−1.18). SPL은 FP32로 남기는 부분을 32채널로 줄이고도 같은 수준(−0.96)을 유지한다.",
    "@IMG fig3_ablation.png|15",
    "#B 어떤 입력에서 취약했나 (사후 분석)",
    "- 보호 없는 INT8(A2m)은 사진 종류(객체 수·주 객체 크기·범주)와 캡션 종류(숫자·색·위치 표현) **전반에서 고르게 −5~−9%p** 떨어졌다. 특정 입력이 아니라 모델 전체에 걸친 이상치 문제라는 원인 분석과 맞는다.",
    "- SPL에서 일부 묶음의 큰 차이(예: 스포츠 사진 −8.4%p)는 select 분할에서 재현되지 않아 우연으로 판단했다. 여러 묶음을 동시에 보면 하나쯤은 우연히 벗어나기 때문이다.",
    "- Hailo는 **주 피사체가 화면의 40% 이상인 사진**에서 두 분할 모두 더 떨어졌다(confirm −4.48 / select −2.72%p). 관찰이며 원인은 확인하지 못했다.",
    "#B ORT와 Hailo의 차이",
    "- Hailo 구성(H-splf)은 select −1.10%p에서 confirm −3.03%p로 떨어졌다. 사후 분석에서 캘리브 묶음만 바꿔도 confirm이 −3.03/−2.35/−1.41%p로 크게 흔들렸고(평균 −2.26), 189장을 모두 써도 −2.04%p였다. **64장 정적 캘리브의 변동**과 select에서 여러 변형 중 하나를 고른 **선택 효과**가 겹친 것으로 본다. 7번 층 전체 16비트(14 컨텍스트)는 3묶음 평균 −1.73%p로 더 안정적이었다.",
    "- ORT는 같은 캘리브로 select −0.78 → confirm −0.96%p로 안정적이었다. 이상치 채널을 FP32로 그대로 두기 때문에 캘리브 범위의 영향을 덜 받는 것으로 해석한다(가설).",
])

# ---------------------------------------------------------------- 7.1
fill_cell(T[16].cell(0, 0), [
    "1. **Hailo 정확도:** Hailo 구성(원본 가중치)이 판정용 분할에서 기준을 넘지 못했다(−3.03%p, 공식 −3.16%p). 64장 캘리브에 따른 변동이 크다(6.6).",
    "2. **Hailo 속도 미측정:** 보드와 HailoRT가 없어 실측하지 못했다. DFC 프로파일러의 컨텍스트별 추정은 전환 비용이 빠져 신뢰할 수 없어 쓰지 않았고, 컨텍스트 수(13)만 확인했다.",
    "3. **텍스트 인코더 HEF 미완:** 공식 설정의 미세조정 단계가 이 PC에서 WSL 전체를 멈춰 중단했다. 현재 텍스트 인코더는 호스트 CPU(FP32)다.",
    "4. **측정 환경:** 속도는 데스크톱 CPU 4스레드 기준이다. 30 fps는 여유 약 2 ms로 겨우 들어온다. 엣지 CPU에서는 절대값이 다르다.",
    "5. **평가 설계:** 자체 분할이라 공개 수치와 직접 비교할 수 없다. 마진은 select를 본 뒤 정했다. confirm은 두 번 쓰였고 NoCaps는 팀 확인 전에 열렸다(6.2). 그래서 최종 구성의 판정은 마지막에 한 번 연 XM3600에 둔다.",
    "6. **NPU 전제로 제외한 더 나은 CPU 구성:** 동적 INT8+fc2 보호(A1-c67-pc)는 CPU에서 더 정확하고 빨랐지만 NPU에서 실행할 수 없어 제외했다.",
    "7. **도메인 적응:** 학습 시드 1개. 중간 동결 이후의 추가 개선(+2.3·+4.3%p, 선택용)은 최종 시험에서 구분되지 않았다(6.6). 텍스트 인코더도 바뀌어 기존 CLIP 색인과 섞어 쓸 수 없다(색인 재생성 필요). 실제 스마트폰 갤러리(셀카·스크린샷 등)에서의 효과는 확인하지 못했다.",
    "8. **Hailo:** 아직 원본 가중치만 이식했다. 최종 구성의 CPU 속도는 같은 그래프의 SPL 측정값으로 갈음했다(적응 모델 직접 측정도 2.30배).",
])

# ---------------------------------------------------------------- 7.2
t = T[17]
add_rows(t, 1)
r72 = [["11/6~11/9", "보드·HailoRT 설치, 공식 HEF와 우리 HEF 구동, 에뮬레이터 임베딩과 일치 확인. 최종 적응 가중치(DA5, 이미지 블록 8~11)를 공식 ONNX에 덮어써 같은 처방으로 재컴파일(예선 기간에 PC에서 미리 시작)", "런타임 버전 불일치 → DFC 5.4.0과 맞는 버전 확인 / 적응 후 Hailo 양자화 손실 재확인(XM3600 포함)"],
       ["11/10~11/16", "예선과 같은 벤치마크·프로토콜로 실측: 배치 1/8 지연·FPS, 메모리·전력, 보드 COCO-confirm 정확도. NPU 활용 지표: CPU 단독 대비 호스트 CPU 부하·전력, HailoRT 도구로 가능한 범위의 NPU 사용률", "컨텍스트 전환 비용으로 FPS가 기대보다 낮을 수 있음"],
       ["11/17~11/23", "Hailo 정확도 개선: 캘리브 사진 확대·여러 묶음 평균으로 선택, 7번 층 전체 16비트(평균 −1.73%p)와 비교해 확정", "정확도와 컨텍스트 수(속도)의 절충"],
       ["11/24~11/30", "갤러리 검색 시연(보드 색인 + 호스트 텍스트 질의), 색인 파이프라인(사진 디코드와 NPU 추론 겹치기·배치) 적용, 텍스트 인코더 HEF 재시도(성공 시 검색 전 과정 NPU), 최종 보고서 제출(11/30)", "시연 안정성, 일정 부족"],
       ["12/1~12/4", "발표 자료, 라이브 시연 리허설, 최종 발표(12/4)", "—"]]
for row, v in zip(t.rows[1:], r72):
    fill_row(row, v)

# ---------------------------------------------------------------- 8
t = T[18]
r8 = {"1": "`[트랙2-예선]_자유_[팀명]_보고서.pdf`",
      "2": "`clip_vitb16_image_fp32.onnx` (344.9 MB) — [[공유 링크]]",
      "3": "최종 `DA5-SPL-b567-k32.onnx` (89.8 MB, ORT) + 적응 가중치 `da5_chosen.pt` (166 MB, 이미지·텍스트 뒤쪽 4블록) / 참고: 원본 가중치 SPL ONNX, HEF `clip_vit_b_16_image_encoder.splf_b7_k32.hailo10h.hef` (79.6 MB) — [[공유 링크]]",
      "4": "ONNX opset 17 · ONNX Runtime 1.23.2 (QDQ) / HEF: Hailo DFC 5.4.0, hailo10h",
      "5": "`reproduce.sh`, `repro_quick.sh`, `repro/da_build.sh`, `src/coco_retrieval.py`, `src/evaluate.py`, `src/benchmark.py`, `src/index_bench.py`, `src/xm3600_eval.py`, `src/nocaps_eval.py`, `src/flickr_eval.py`, `src/cc3m_eval.py` / 학습: `src/da_*.py`",
      "6": "`requirements.lock.txt` (버전 고정), `requirements.txt`",
      "7": "`README.md`, `docs/06_재현가이드.md`",
      "8": "공식 URL에서 자동 다운로드: `src/download_data.py`(COCO val2017·캡션, CIFAR-100), `src/da_data.py`·`src/cc3m_eval.py`(학습 재현 시 train2017 11.8만·CC3M 10만 장), `src/xm3600_prep.py`(XM3600), `src/nocaps_prep.py`, `src/flickr_eval.py`"}
for row in t.rows[1:]:
    k = row.cells[0].text.strip()
    if k in r8:
        fill_row(row, [None, None, r8[k], None])

# ---------------------------------------------------------------- 9
fill_cell(T[19].cell(0, 0), [
    "Windows 10/11 + WSL2 Ubuntu 22.04, 메모리 16 GB 이상 권장, GPU 불필요.",
    "```# PowerShell(관리자): WSL 설치 후 재부팅",
    "```wsl --install -d Ubuntu-22.04",
    "```# Ubuntu: 코드 풀기와 파이썬 환경",
    "```sudo apt update && sudo apt install -y python3.10-venv unzip",
    "```cd ~ && unzip /mnt/c/Users/<사용자>/Downloads/halio_repro_<날짜>.zip && cd ~/halio_project",
    "```python3 -m venv ~/venvs/halio && source ~/venvs/halio/bin/activate",
    "```pip install --upgrade pip",
    "```pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cpu",
    "```export HALIO_DATA_DIR=/mnt/d/halio_data      # 데이터·모델 저장 위치 (선택)",
    "```SELFTEST=1 bash run_pilot.sh                 # 2~3분 점검 (가짜 데이터)",
])
fill_cell(T[20].cell(0, 0), [
    "```# 전체: 베이스라인(FP32)과 최적화 모델 모두, 정확도 + 속도 + 기대값 비교",
    "```bash reproduce.sh",
    "```SKIP_BENCH=1 bash reproduce.sh               # 정확도만",
    "```# 개별 명령 (M=$(python -m src.common models-dir))",
    "```# 베이스라인 FP32",
    "```python -m src.coco_retrieval eval --model $M/clip_vitb16_image_fp32.onnx --code RP-A0 --split confirm --i-know-final",
    "```python -m src.evaluate --model $M/clip_vitb16_image_fp32.onnx --code RP-A0 --quant-mode fp32 --eval-set test --i-know-final --threads 4",
    "```# 최적화 모델 (최종 DA5-SPL): 적응 가중치 → FP32 ONNX + 적응 텍스트 임베딩 → SPL 처방",
    "```bash repro/da_build.sh; D=$(python -m src.common store-dir)/models_da5",
    "```HALIO_MODELS_SUBDIR=models_da5 python -m src.coco_retrieval eval --model $D/imp/SPL-b567-k32.onnx --code RP-DA5-SPL --split confirm --i-know-final",
    "```HALIO_MODELS_SUBDIR=models_da5 python -m src.evaluate --model $D/imp/SPL-b567-k32.onnx --code RP-DA5-SPL --quant-mode static --eval-set test --i-know-final --threads 4",
    "```# 최종 시험 XM3600 (다운로드 약 0.9 GB)",
    "```python -m src.xm3600_prep download && python -m src.xm3600_eval text --tag da5 --text-model $D/text_state.pt",
    "```python -m src.xm3600_eval eval --i-know-final RP-DA5-SPL=$D/imp/SPL-b567-k32.onnx:da5",
    "```# 속도·메모리, 색인 실측",
    "```python -m src.benchmark --threads 4 --out-csv results/latency_repro.csv --models RP-A0=$M/clip_vitb16_image_fp32.onnx RP-DA5-SPL=$D/imp/SPL-b567-k32.onnx",
    "```python -m src.index_bench run",
])
fill_cell(T[21].cell(0, 0), [
    "`reproduce.sh` 마지막에 아래 표가 나온다(빈 저장 폴더에서 실제로 나온 출력, 속도 줄은 다른 PC 실측).",
    "```구성           항목                    제출값   재현값  판정",
    "```A0             COCO-confirm R@1        41.79%   41.79%  일치",
    "```A0             CIFAR-100 test top-1    68.01%   68.01%  일치",
    "```SPL-b567-k32   COCO-confirm R@1        40.84%   40.84%  일치",
    "```SPL-b567-k32   CIFAR-100 test top-1    68.66%   68.66%  일치",
    "```DA5-SPL        COCO-confirm R@1        56.97%   56.97%  일치",
    "```DA5-SPL        CIFAR-100 test top-1    68.16%   68.16%  일치      (4구성 × R@1/5/10·CIFAR = 16행)",
    "```RP-SPL-b567-k32    34.81 ms   FP32 대비 2.37배   (다른 PC 실측. 속도는 CPU마다 다름)",
    "```종합: 정확도 재현 성공",
    "- **대응:** CIFAR test → 6.5 정확도 지표 2. 속도 배율 → 6.5 Latency(원 PC 2.28배). `results/coco_ret_confirm_repro.md`의 신뢰구간 표 → 6.5 정확도 지표 1·3과 그림 1.",
    "- **R@1 표기 차이:** 재현 표의 R@1(41.79%, 56.97%)은 캡션 단위 평균, 보고서 표(41.80%, 56.97%)는 이미지별 성적의 평균이다. 같은 결과를 두 방식으로 계산한 것이다.",
    "- **판정 기준:** 정확도는 제출값과 ±0.2%p 안이면 일치. 속도는 절대값이 아니라 FP32 대비 배율로 본다.",
])
fill_cell(T[22].cell(0, 0), [
    "- **최초 실행 2~4시간:** 데이터 약 1.2 GB 다운로드(10~40분), CLIP 가중치는 open_clip이 자동으로 받는다. 끊겨도 다시 실행하면 끝난 단계는 건너뛴다.",
    "- **저장 공간:** 데이터·모델 약 4~5 GB(`HALIO_DATA_DIR`), 파이썬 환경 약 2 GB.",
    "- **torch는 CPU판(2.14.1+cpu):** `--extra-index-url`을 빼면 GPU판(약 6 GB)을 받거나 설치에 실패한다.",
    "- **속도 측정:** 다른 프로그램을 끄고 Windows 절전을 해제한다. 절대값은 CPU마다 다르다.",
    "- **공유한 모델 파일 사용:** `$M/`과 `$M/imp/`(최종은 `models_da5/imp/`)에 같은 이름으로 두면 양자화 단계를 건너뛴다.",
    "- **적응 가중치:** GitHub Release의 `da5_chosen.pt`를 `repro/weights/`에 두면 자동으로 쓰고 sha256을 확인한다. `DA_TRAIN=1`로 직접 학습하면 CPU 연산 차이로 값이 아주 조금 다를 수 있다(약 15시간, 메모리 23 GB 안팎, C 드라이브 대신 넉넉한 디스크 권장).",
    "- **Hailo HEF:** 재컴파일에는 Hailo DFC 5.4.0이 필요하다(Hailo 개발자 사이트 로그인, 재배포 불가). ORT 수치 재현에는 필요 없다.",
    "- **자주 막히는 곳:** WSL 사용자 이름은 영어 소문자로 시작해야 한다. 윈도우 경로(`C:\\...`)는 `/mnt/c/...`로 쓴다. 압축이 풀린 폴더 이름은 `halio_project`다.",
])

# ---------------------------------------------------------------- references
last_box = T[22]._tbl
h1 = None
for p in d.paragraphs:
    if p.style.name == "Heading 1":
        h1 = p
ref_h = Paragraph(copy.deepcopy(h1._p), h1._parent)
for r in ref_h._p.findall(qn("w:r")):
    ref_h._p.remove(r)
sp = last_box.getnext()
sp.addnext(ref_h._p)
rr = ref_h.add_run("참고 문헌"); rr.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
refs = ["[1] Immich 문서, Searching. https://docs.immich.app/features/searching",
        "[2] Hailo Model Zoo, Hailo-10H Zero-shot Classification. https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_zero_shot_classification.rst",
        "[3] Y. Bondarenko, M. Nagel, T. Blankevoort, Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing, NeurIPS 2023. https://arxiv.org/abs/2306.12929",
        "[4] N. Jiang, A. Dravid, A. Efros, Y. Gandelsman, Vision Transformers Don't Need Trained Registers, NeurIPS 2025. https://arxiv.org/abs/2506.08010",
        "[5] MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training, CVPR 2024. https://arxiv.org/abs/2311.17049",
        "[6] Activation Quantization of Vision Encoders Needs Prefixing Registers, ICML 2026 Workshop (AdaptFM). https://arxiv.org/abs/2510.04547",
        "[7] A. Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP), ICML 2021. https://arxiv.org/abs/2103.00020",
        "[8] M. Wortsman et al., Robust fine-tuning of zero-shot models (WiSE-FT), CVPR 2022. https://arxiv.org/abs/2109.01903",
        "[9] S. Goyal et al., Finetune Like You Pretrain: Improved Finetuning of Zero-Shot Vision Models (FLYP), CVPR 2023. https://arxiv.org/abs/2212.00638",
        "[10] P. Young, A. Lai, M. Hodosh, J. Hockenmaier, From image descriptions to visual denotations (Flickr30k), TACL 2014. https://aclanthology.org/Q14-1006/",
        "[11] H. Agrawal et al., nocaps: novel object captioning at scale, ICCV 2019. https://arxiv.org/abs/1812.08658",
        "[12] A. V. Thapliyal et al., Crossmodal-3600: A Massively Multilingual Multimodal Evaluation Dataset, EMNLP 2022. https://arxiv.org/abs/2205.12522",
        "[13] P. Sharma et al., Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning, ACL 2018. https://aclanthology.org/P18-1238/"]
prev = ref_h
normal_like = None
for p in d.paragraphs:
    if p.style.name == "Normal" and p.text == "":
        normal_like = p
        break
for txt in refs:
    np_ = Paragraph(copy.deepcopy(normal_like._p), normal_like._parent)
    for r in np_._p.findall(qn("w:r")):
        np_._p.remove(r)
    prev._p.addnext(np_._p)
    add_runs(np_, txt, 8.5)
    np_.paragraph_format.space_after = Pt(2)
    prev = np_

d.save("report_draft.docx")
print("saved")
