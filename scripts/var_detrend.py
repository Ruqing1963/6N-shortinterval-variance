import numpy as np, json, time
import os
_HERE=os.path.dirname(os.path.abspath(__file__))
_DATA=os.path.normpath(os.path.join(_HERE,"..","data"))
_FIG=os.path.normpath(os.path.join(_HERE,"..","figures"))
os.makedirs(_DATA,exist_ok=True); os.makedirs(_FIG,exist_ok=True)
t0=time.time()
LIMIT=600_000_000; NMAX=LIMIT//6
sieve=np.ones(LIMIT+3,dtype=bool); sieve[:2]=False
for p in range(2,int((LIMIT+2)**0.5)+1):
    if sieve[p]: sieve[p*p::p]=False
is_twin=np.zeros(NMAX+1,dtype=bool)
for a in range(1,NMAX+1,10_000_000):
    b=min(a+10_000_000,NMAX+1); idx=np.arange(a,b,dtype=np.int64)
    is_twin[a:b]=sieve[6*idx-1]&sieve[6*idx+1]
del sieve
ntwin=int(is_twin.sum()); p0=ntwin/NMAX
print("ntwin=%d p0=%.7f (%.1fs)"%(ntwin,p0,time.time()-t0))

# prefix sum of indicator (centres 1..NMAX -> index 1..NMAX)
cs=np.zeros(NMAX+1,dtype=np.int64)
np.cumsum(is_twin[1:NMAX+1],out=cs[1:])
del is_twin
print("cumsum done %.1fs"%(time.time()-t0))

W=1_000_000; half=W//2     # smoothing scale for local density

Hs=[10,20,30,50,70,100,150,200,300,400,500,700,1000,1500,2000,3000,5000,7000,
    10000,15000,20000,30000,50000,100000]
rows=[]
print("\n   H     F_total   F_local(detrended)  F_trend   trend%")
for H in Hs:
    m=NMAX//H
    a=np.arange(0,m,dtype=np.int64)*H            # window starts; window = (a, a+H]
    o=(cs[a+H]-cs[a]).astype(np.float64)         # observed counts
    c=a+H//2                                      # window centre
    chi=np.minimum(c+half,NMAX); clo=np.maximum(c-half,0)
    rho=(cs[chi]-cs[clo]).astype(np.float64)/(chi-clo)   # local smooth density
    e=rho*H                                       # smooth-trend expected count
    mu=o.mean()
    Ft=float(o.var()/mu)
    Floc=float((o-e).var()/mu)
    Ftr=float(e.var()/mu)
    rows.append(dict(H=int(H),F_total=Ft,F_local=Floc,F_trend=Ftr))
    print("%7d  %.4f     %.4f            %.4f    %.0f%%"%(H,Ft,Floc,Ftr,100*Ftr/Ft))

json.dump(dict(NMAX=NMAX,LIMIT=LIMIT,p0=p0,ntwin=ntwin,W=W,detrend=rows),
          open(os.path.join(_DATA,"var_detrend.json"),"w"))
print("\nTOTAL %.1fs"%(time.time()-t0))
