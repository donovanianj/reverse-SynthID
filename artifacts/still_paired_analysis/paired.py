import numpy as np, cv2, subprocess
S="/tmp/claude-0/-home-user-reverse-SynthID/b7b126f1-18e5-5ce9-b2fa-a45ca795acac/scratchpad/"
U="/root/.claude/uploads/b7b126f1-18e5-5ce9-b2fa-a45ca795acac/"
st=cv2.imread(S+"stills/still_user.jpg").astype(np.float32)
raw=subprocess.run(["ffmpeg","-v","error","-i",U+"61a8260b-1st-2026_Xu_Ning_e42800_f27376.mp4","-vframes","8","-f","rawvideo","-pix_fmt","bgr24","-"],capture_output=True).stdout
fr=np.frombuffer(raw,np.uint8).reshape(-1,1072,1920,3).astype(np.float32); f6=fr[6]
g=lambda a:cv2.cvtColor(a.astype(np.float32),cv2.COLOR_BGR2GRAY)
# subpixel alignment: still (1918 wide) vs frame crop
ref=g(f6)[:,1:1919]; mov=g(st)
win=cv2.createHanningWindow(mov.shape[::-1],cv2.CV_32F)
(dx,dy),resp=cv2.phaseCorrelate(ref,mov,win); print("subpixel shift still vs frame6[:,1:1919]: dx %.3f dy %.3f resp %.3f"%(dx,dy,resp))
# also scale check: compare via log-polar? quick check of resize: test zoom factors
best=None
for s in [0.995,0.998,1.0,1.002,1.005]:
    M=np.float32([[s,0,(1-s)*959],[0,s,(1-s)*536]]); w=cv2.warpAffine(ref,M,(1918,1072),flags=cv2.INTER_CUBIC)
    c=np.corrcoef(w[50:-50,50:-50].ravel(),mov[50:-50,50:-50].ravel())[0,1]; print(" scale",s,"corr",round(c,5))
M=np.float32([[1,0,dx],[0,1,dy]]); refA=cv2.warpAffine(f6[:,1:1919],M,(1918,1072),flags=cv2.INTER_CUBIC)
# per-channel gain/offset match (linear regression)
D=np.zeros_like(st)
for c in range(3):
    a,b=np.polyfit(refA[20:-20,20:-20,c].ravel(),st[20:-20,20:-20,c].ravel(),1); D[...,c]=st[...,c]-(a*refA[...,c]+b); print(" ch",c,"gain %.4f off %.2f"%(a,b), "resid sd %.2f"%D[20:-20,20:-20,c].std())
np.save(S+"D_still_minus_f6.npy",D[20:-20,20:-20]); np.save(S+"f6_aligned.npy",refA)
d8=lambda a:np.clip(a*8+128,0,255).astype(np.uint8)
cv2.imwrite(S+"D_vis.png",d8(D)); cv2.imwrite(S+"D_vis_crop.png",cv2.resize(d8(D[400:656,800:1056]),None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
