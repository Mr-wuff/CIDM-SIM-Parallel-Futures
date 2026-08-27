import numpy as np

def make_demo_history(hist_len=4,channels=1,size=48):
    yy,xx=np.mgrid[0:size,0:size]/float(size)
    hist=[]
    for t in range(hist_len):
        phase=0.25*t
        field=(np.exp(-((xx-(0.35+0.02*t))**2+(yy-0.45)**2)/0.02)
               -0.55*np.exp(-((xx-0.68)**2+(yy-(0.60-0.01*t))**2)/0.03)
               +0.12*np.sin(2*np.pi*(2*xx+yy+phase))).astype("float32")
        hist.append(field[None,...])
    return np.stack(hist,axis=0)[None,...]
