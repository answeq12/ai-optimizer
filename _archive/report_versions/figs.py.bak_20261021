import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fp="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(fp); name=fm.FontProperties(fname=fp).get_name()
plt.rcParams.update({"font.family":name,"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.unicode_minus":True})
ACC="#2563eb"; G="#9ca3af"; DK="#374151"

# Fig1 Pareto
pts=[("FP32 (A0)",58.06,0,0,0,"base"),("동적 INT8 (A1)",22.25,-1.48,-2.18,-0.78,"ref"),
     ("정적+6·7 fc2 FP32+채널별 (S-c67-pc)",26.30,-0.86,-1.51,-0.22,"cand"),
     ("채택 처방: 이상치 채널 분리 (SPL-b567-k32)",25.52,-0.96,-1.56,-0.35,"final"),
     ("동적+6·7 fc2 FP32 (A1-c67-pc, NPU 불가)",24.05,-0.57,-1.02,-0.12,"ref")]
fig,ax=plt.subplots(figsize=(6.3,3.3),dpi=200)
ax.axhline(-2,ls="--",color=G,lw=1); ax.text(59.5,-2.08,"마진 −2%p",ha="right",va="top",color=DK,fontsize=8)
ax.axhline(0,color="#d1d5db",lw=0.8)
for lab,x,y,lo,hi,k in pts:
    c=ACC if k=="final" else (DK if k=="base" else G)
    face=c if k!="ref" else "white"
    ax.errorbar(x,y,yerr=[[y-lo],[hi-y]] if k!="base" else None,fmt="o",color=c,mfc=face,mec=c,ms=6,capsize=2,lw=1.2)
ly={"동적+6·7 fc2 FP32 (A1-c67-pc, NPU 불가)":-0.35,"정적+6·7 fc2 FP32+채널별 (S-c67-pc)":-0.75,"채택 처방: 이상치 채널 분리 (SPL-b567-k32)":-1.15,"동적 INT8 (A1)":-1.6}
for lab,x,y,*_ in pts:
    if lab=="FP32 (A0)":
        ax.text(x-1,y+0.15,lab,fontsize=7.5,ha="right",color=DK); continue
    fin="채택" in lab
    ax.annotate(lab,xy=(x,y),xytext=(32,ly[lab]),fontsize=7.5,va="center",color=ACC if fin else DK,fontweight="bold" if fin else "normal",
                arrowprops=dict(arrowstyle="-",color="#d1d5db",lw=0.7,shrinkA=2,shrinkB=4))
ax.set_xlim(15,62); ax.set_ylim(-2.6,0.6)
ax.set_xlabel("인코더 지연 p50 (ms, CPU 4스레드·배치 1, 낮을수록 좋음)"); ax.set_ylabel("COCO-confirm R@1 차이 (%p, FP32 대비)")
ax.set_title("양자화 처방 비교(원본 가중치): SPL은 2.28배 빠르면서 R@1 −0.96%p",fontsize=10,loc="left",fontweight="bold")
fig.tight_layout(); fig.savefig("fig1_pareto.png")

# Fig2 mechanism
blocks=list(range(12)); lev=[43.2,59.3,45.6,63.9,44.1,18.0,12.2,8.9,57.9,49.7,47.3,70.7]; amax=[7.8,5.4,5.8,4.3,6.6,17.4,26.5,42.3,8.9,14.9,16.8,5.9]
fig,ax=plt.subplots(figsize=(6.3,2.8),dpi=200)
cols=[ACC if b in (5,6,7) else G for b in blocks]
ax.bar(blocks,lev,color=cols,width=0.65)
for b,l,a in zip(blocks,lev,amax):
    ax.text(b,l+1.5,f"{l:.0f}칸",ha="center",fontsize=7.5,color=DK)
    if b in (5,6,7): ax.text(b,l+8,f"최대 {a:.0f}",ha="center",fontsize=7,color=ACC)
ax.set_xticks(blocks,[f"b{b}" for b in blocks]); ax.set_ylim(0,82)
ax.set_ylabel("99.9% 값이 쓰는 8비트 눈금 수 (255칸 중)"); ax.set_xlabel("트랜스포머 블록 (fc2 입력)")
ax.set_title("블록 5~7 fc2 입력은 소수 이상치 때문에 8비트 눈금을 9~18칸만 사용",fontsize=10,loc="left",fontweight="bold")
fig.tight_layout(); fig.savefig("fig2_mechanism.png")

# Fig3 ablation
rows=[("보호 없는 정적 INT8 (A2m)",-6.89,-8.00,-5.82,0),("+ 6·7번 fc2를 FP32로",-1.18,-1.85,-0.50,0),
      ("+ 채널별 가중치 양자화",-0.86,-1.51,-0.22,0),("채택: 이상치 32채널 분리 (SPL)",-0.96,-1.56,-0.35,1)]
fig,ax=plt.subplots(figsize=(6.8,2.5),dpi=200)
for i,(lab,d,lo,hi,f) in enumerate(rows):
    y=len(rows)-1-i; c=ACC if f else G
    ax.plot([lo,hi],[y,y],color=c,lw=2.2); ax.plot(d,y,"o",color=c,ms=7)
    ax.text(d,y+0.28,f"{d:.2f}".replace("-","−"),ha="center",fontsize=8,color=DK)
ax.set_yticks(range(len(rows)),[r[0] for r in rows][::-1])
for t in ax.get_yticklabels():
    if "채택" in t.get_text(): t.set_color(ACC); t.set_fontweight("bold")
ax.axvline(0,color=DK,lw=1); ax.axvline(-2,color=G,ls="--",lw=1); ax.text(-2.1,len(rows)-0.45,"마진 −2%p",ha="right",fontsize=7.5,color=DK)
ax.set_xlim(-8.6,0.6); ax.set_ylim(-0.6,len(rows)-0.1)
ax.set_xlabel("COCO-confirm R@1 차이 (%p, FP32 대비, 95% 범위)")
ax.set_title("6·7번 fc2 보호 한 단계가 손실 대부분을 회복 (−6.89 → −1.18%p)",fontsize=10,loc="left",fontweight="bold")
fig.tight_layout(); fig.savefig("fig3_ablation.png")
# Fig4 decomposition
labs=["원본 FP32\n(기준 모델)","원본 + SPL INT8","적응 FP32","적응 + SPL INT8\n(최종)"]
coco=[41.80,40.85,54.36,53.50]; fl=[61.52,59.94,73.60,72.48]
import numpy as np
x=np.arange(4); w=0.36
fig,ax=plt.subplots(figsize=(6.6,2.9),dpi=200)
cols=[DK,G,"#93c5fd",ACC]
b1=ax.bar(x-w/2,coco,w,color=cols,edgecolor="white"); b2=ax.bar(x+w/2,fl,w,color=cols,alpha=0.55,edgecolor="white",hatch="//")
for xi,v in zip(x-w/2,coco): ax.text(xi,v+1,f"{v:.1f}",ha="center",fontsize=7.5,color=DK)
for xi,v in zip(x+w/2,fl): ax.text(xi,v+1,f"{v:.1f}",ha="center",fontsize=7.5,color=DK)
ax.set_xticks(x,labs,fontsize=8); ax.set_ylim(0,85); ax.set_ylabel("텍스트→이미지 R@1 (%)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=G,label="COCO-confirm (자체 판정 분할)"),Patch(facecolor=G,alpha=0.55,hatch="//",label="Flickr30k 1K test (학습·선택에 안 씀)")],fontsize=7.5,loc="upper left",frameon=False)
ax.set_title("도메인 적응 +12.6%p, 양자화 −0.9%p → 최종은 원본보다 +11.7%p (Flickr +11.0%p)",fontsize=9.5,loc="left",fontweight="bold")
fig.tight_layout(); fig.savefig("fig4_da.png")
print("ok")
