import sys, numpy as np, cv2, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"src/extraction")
from synthid_bypass_v4 import SpectralCodebookV4
S="/tmp/claude-0/-home-user-reverse-SynthID/b7b126f1-18e5-5ce9-b2fa-a45ca795acac/scratchpad/"
cb=SpectralCodebookV4(); cb.load("artifacts/spectral_codebook_v4.npz")
keys=list(cb.profiles.keys()); print("profiles:",keys)
rng=np.random.default_rng(0)
st=cv2.imread(S+"stills/still_user.jpg")[...,::-1].astype(np.float32)
f6=np.load(S+"f6_aligned.npy")[...,::-1].astype(np.float32)
ctrl=cv2.imread(S+"f6_q90.jpg")[...,::-1].astype(np.float32)
R=np.load(S+"R_still.npy")[...,::-1].astype(np.float32); Rc=np.load(S+"R_ctrl.npy")[...,::-1].astype(np.float32)
veo=cv2.imread(S+"stills/veo1080_remake_t3.png")[...,::-1].astype(np.float32)
imgs={"still":st,"frame6":f6,"frame6_jpeg":ctrl,"resid_still":R,"resid_ctrl":Rc,"veo_remake_frame":veo}
def score(img,prof,top_k=128,floor=0.75):
    pH,pW=prof.shape
    work=cv2.resize(img,(pW,pH),interpolation=cv2.INTER_AREA).astype(np.float64)
    out=[]
    for ch in range(3):
        cons=prof.consensus_coherence[:,:,ch].copy(); cons[0,0]=0
        cand=np.argsort(cons.ravel())[-top_k:]; cand=cand[cons.ravel()[cand]>=floor]
        if len(cand)==0: out.append((np.nan,np.nan,0)); continue
        ph=np.angle(np.fft.fft2(work[:,:,ch])).ravel()
        m=1-np.abs(np.angle(np.exp(1j*(ph[cand]-prof.consensus_phase[:,:,ch].ravel()[cand]))))/np.pi
        z=(m.mean()-0.5)/(0.2887/np.sqrt(len(m)))
        out.append((m.mean(),z,len(m)))
    return out
print(f"{'image':18s} {'profile':42s}  R-match(z)   G-match(z)   B-match(z)")
res={}
for k in keys:
    prof=cb.profiles[k]
    for nm,im in imgs.items():
        o=score(im,prof); res[(nm,k)]=o
for nm in imgs:
    for k in keys:
        o=res[(nm,k)]
        print(f"{nm:18s} {k[0][:24]+'|'+str(k[1])+'x'+str(k[2]):42s}  "+"  ".join(f"{m:.3f}({z:+5.1f})" for m,z,n in o))
    print()
