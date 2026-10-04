import numpy as np, cv2
S="/tmp/claude-0/-home-user-reverse-SynthID/b7b126f1-18e5-5ce9-b2fa-a45ca795acac/scratchpad/"
st=cv2.imread(S+"stills/still_user.jpg").astype(np.float32)
f6=np.load(S+"f6_aligned.npy")
# control: frame6 aligned -> JPEG q=90 (similar quant tables)
cv2.imwrite(S+"f6_q90.jpg",np.clip(f6,0,255).astype(np.uint8),[cv2.IMWRITE_JPEG_QUALITY,90])
ctrl=cv2.imread(S+"f6_q90.jpg").astype(np.float32)
q=__import__("PIL.Image",fromlist=["x"]).open(S+"f6_q90.jpg").quantization; print("ctrl quant",{k:list(v)[:8] for k,v in q.items()})
def feats(x):
    x=x/255.0; F=[np.ones_like(x[...,0])]
    for c in range(3): F.append(x[...,c])
    for a in range(3):
        for b in range(a,3): F.append(x[...,a]*x[...,b])
    for c in range(3): F.append(x[...,c]**3)
    for s in (0.7,1.5,3,6):
        for c in range(3):
            F.append(x[...,c]-cv2.GaussianBlur(x[...,c],(0,0),s))
    return np.stack(F,-1)
def residual(target,src):
    X=feats(src)[24:-24,24:-24]; Y=target[24:-24,24:-24]
    Xf=X.reshape(-1,X.shape[-1]); R=np.zeros(Y.shape,np.float32)
    sub=np.random.default_rng(0).choice(len(Xf),400000,replace=False)
    for c in range(3):
        beta,*_=np.linalg.lstsq(Xf[sub],Y[...,c].reshape(-1)[sub],rcond=None)
        R[...,c]=(Y[...,c].reshape(-1)-Xf@beta).reshape(Y.shape[:2])
    return R
Rs=residual(st,f6); Rc=residual(ctrl,f6)
print("residual sd still:",Rs.std((0,1)).round(2)," ctrl(JPEG only):",Rc.std((0,1)).round(2))
np.save(S+"R_still.npy",Rs); np.save(S+"R_ctrl.npy",Rc)
