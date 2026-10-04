import subprocess, numpy as np, cv2, sys
S="/tmp/claude-0/-home-user-reverse-SynthID/b7b126f1-18e5-5ce9-b2fa-a45ca795acac/scratchpad/"
U="/root/.claude/uploads/b7b126f1-18e5-5ce9-b2fa-a45ca795acac/"
H,W=1080,1920; win=np.outer(np.hanning(H),np.hanning(W)).astype(np.float32)
def unit(path,h,dys):
    raw=subprocess.run(["ffmpeg","-v","error","-i",path,"-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True).stdout
    fr=np.frombuffer(raw,np.uint8).reshape(-1,h,W,3); acc={d:0 for d in dys}
    for f in fr:
        f=f.astype(np.float32); y=0.299*f[...,0]+0.587*f[...,1]+0.114*f[...,2]
        r=y-cv2.GaussianBlur(y,(0,0),2.0)
        for d in dys:
            rr=cv2.copyMakeBorder(r,d,(H-h)-d,0,0,cv2.BORDER_CONSTANT,value=0) if h<H else r
            F=np.fft.rfft2(rr*win); acc[d]=acc[d]+F/np.maximum(np.abs(F),1e-9)
    return {d:(a/len(fr)).astype(np.complex64) for d,a in acc.items()}, len(fr)
if sys.argv[1]=="remake":
    u,n=unit(U+"bd3ec56d-Generated_Video_October_04_2026_-_12_26PM.mp4",1080,[0]); np.save(S+"U1080_remake.npy",u[0]); print("remake",n)
else:
    u,n=unit(U+"61a8260b-1st-2026_Xu_Ning_e42800_f27376.mp4",1072,list(range(9)))
    for d,a in u.items(): np.save(S+f"U1080_orig_dy{d}.npy",a)
    print("orig",n)
