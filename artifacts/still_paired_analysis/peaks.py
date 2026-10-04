import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, maximum_filter
S="/tmp/claude-0/-home-user-reverse-SynthID/b7b126f1-18e5-5ce9-b2fa-a45ca795acac/scratchpad/"
out={}
for nm in ["still","ctrl"]:
    R=np.load(S+f"R_{nm}.npy"); H,W,_=R.shape
    win=np.outer(np.hanning(H),np.hanning(W))
    for ch,lab in [((0.114,0.587,0.299),"Y"),((0,1,0),"G")]:
        y=R[...,0]*ch[0]+R[...,1]*ch[1]+R[...,2]*ch[2]
        M=np.abs(np.fft.fftshift(np.fft.fft2(y*win)))
        L=np.log(M+1e-6); bg=median_filter(L,size=15); P=L-bg   # peakiness
        out[(nm,lab)]=(M,P)
        cy,cx=H//2,W//2; yy,xx=np.mgrid[:H,:W]; r=np.hypot((yy-cy)/H,(xx-cx)/W)
        m=(P==maximum_filter(P,size=7))&(r>0.01)&(xx>=cx)
        idx=np.argsort(-P[m])[:25]; ys,xs=np.nonzero(m)
        pk=[(int(ys[i]-cy),int(xs[i]-cx),round(float(P[ys[i],xs[i]]),2)) for i in idx]
        print(nm,lab,H,W,"top peaks (ky,kx,log-excess):",pk[:20])
        print("   #peaks excess>2.0:",int(((P>2.0)&(r>0.01)).sum()), " >1.5:",int(((P>1.5)&(r>0.01)).sum()))
fig,ax=plt.subplots(1,2,figsize=(18,5.5))
for a,nm in zip(ax,["still","ctrl"]):
    P=out[(nm,"Y")][1]; a.imshow(np.clip(P,0,3),cmap="magma"); a.set_title(f"{nm}: residual spectrum peak excess (log |F| - local median)"); a.axis("off")
fig.tight_layout(); fig.savefig(S+"resid_peaks.png",dpi=90)
