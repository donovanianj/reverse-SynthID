import numpy as np
S="/tmp/claude-0/-home-user-reverse-SynthID/b7b126f1-18e5-5ce9-b2fa-a45ca795acac/scratchpad/"
H,W=1080,1920; Wr=W//2+1
A=np.load(S+"U1080_remake.npy")
ky=np.round(np.fft.fftfreq(H)*H).astype(int)[:,None]; kx=np.arange(Wr)[None,:]
fy,fx=ky/H,kx/W; rad=np.hypot(fy,fx)
def near(k,step,tol=2): return np.abs(k-np.round(k/step)*step)<=tol
grid=near(ky,H/16)&near(kx,W/16)
band=(rad>=0.08)&(rad<=0.45)&~grid&((kx>0)|(ky>0))
ca=np.abs(A)
print("remake coh median in band",np.median(ca[band]).round(3))
res={}
for d in range(9):
    B=np.load(S+f"U1080_orig_dy{d}.npy"); cb=np.abs(B)
    q=np.real(A*np.conj(B)); qn=np.real(A*np.conj(np.roll(B,(9,13),(0,1))))
    both=(ca>0.3)&(cb>0.15)&band
    cos=np.cos(np.angle(A)-np.angle(B))
    n=both.sum(); mc=cos[both].mean()
    bothn=(ca>0.3)&(np.roll(cb,(9,13),(0,1))>0.15)&band; cosn=np.cos(np.angle(A)-np.angle(np.roll(B,(9,13),(0,1))))
    print(f"dy={d}: bins coherent in both {n:5d}, mean cos {mc:+.3f}, frac>0 {np.mean(cos[both]>0):.3f} | shifted-null: n {bothn.sum()} mean cos {cosn[bothn].mean():+.3f}  | sum Q band {q[band].sum():+.1f} null {qn[band].sum():+.1f}")
    res[d]=(mc,n)
print("orig coh median in band",np.median(cb[band]).round(3),"floor",round(np.sqrt(np.pi/(4*193)),3))
