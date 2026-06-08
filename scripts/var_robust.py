import numpy as np, json, time
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
cs=np.zeros(NMAX+1,dtype=np.int64); np.cumsum(is_twin[1:NMAX+1],out=cs[1:]); del is_twin
print("ready %.1fs  p0=%.7f"%(time.time()-t0,p0))

def Flocal(H,W):
    half=W//2; m=NMAX//H
    a=np.arange(0,m,dtype=np.int64)*H
    o=(cs[a+H]-cs[a]).astype(np.float64)
    c=a+H//2; chi=np.minimum(c+half,NMAX); clo=np.maximum(c-half,0)
    rho=(cs[chi]-cs[clo]).astype(np.float64)/(chi-clo)
    return float((o-rho*H).var()/o.mean())

Hs=[100,300,1000,3000,10000,30000]
print("\n   H    W=5e5   W=1e6   W=2e6   W=4e6")
for H in Hs:
    vals=[Flocal(H,W) for W in (500_000,1_000_000,2_000_000,4_000_000)]
    print("%7d  %.4f  %.4f  %.4f  %.4f"%(H,*vals))
print("\nTOTAL %.1fs"%(time.time()-t0))
