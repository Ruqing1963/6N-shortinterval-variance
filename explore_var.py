import numpy as np, time
import os
_HERE=os.path.dirname(os.path.abspath(__file__))
_DATA=os.path.normpath(os.path.join(_HERE,"..","data"))
_FIG=os.path.normpath(os.path.join(_HERE,"..","figures"))
os.makedirs(_DATA,exist_ok=True); os.makedirs(_FIG,exist_ok=True)
t0=time.time()
LIMIT=200_000_000; NMAX=LIMIT//6
# prime sieve
sieve=np.ones(LIMIT+3,dtype=bool); sieve[:2]=False
for p in range(2,int((LIMIT+2)**0.5)+1):
    if sieve[p]: sieve[p*p::p]=False
# twin centre indicator over N=1..NMAX
is_twin=np.zeros(NMAX+1,dtype=bool)
for a in range(1,NMAX+1,10_000_000):
    b=min(a+10_000_000,NMAX+1); idx=np.arange(a,b,dtype=np.int64)
    is_twin[a:b]=sieve[6*idx-1]&sieve[6*idx+1]
del sieve
ntwin=int(is_twin.sum()); p0=ntwin/NMAX
print("NMAX=%d  ntwin=%d  p0=%.7f  (%.1fs)"%(NMAX,ntwin,p0,time.time()-t0))

# ---- Fano factor F(H)=Var/Mean over disjoint windows ----
x=is_twin[1:NMAX+1].astype(np.int64)        # length NMAX, centres 1..NMAX
cs=np.concatenate([[0],np.cumsum(x)])       # prefix sums, len NMAX+1
print("cumsum done %.1fs"%(time.time()-t0))

Hs=[5,10,20,30,50,70,100,150,200,300,500,700,1000,1500,2000,3000,5000,7000,10000,15000,20000,30000]
print("\n   H      m windows     mean        var       Fano F     F-1")
fano=[]
for H in Hs:
    m=NMAX//H
    ends=np.arange(H,(m+1)*H,H)             # H,2H,...,mH  (length m)
    counts=cs[ends]-cs[ends-H]
    mu=counts.mean(); var=counts.var()
    F=var/mu
    fano.append((H,m,mu,var,F))
    print("%6d  %9d   %9.4f  %9.4f   %.5f   %+.5f"%(H,m,mu,var,F,F-1))

# ---- pair correlation C(k) for k=1..Kmax ----
Kmax=2000
A=np.zeros(Kmax+1,dtype=np.int64)
xb=is_twin[1:NMAX+1]                          # bool view
for k in range(1,Kmax+1):
    A[k]=np.count_nonzero(xb[:-k]&xb[k:])
print("\npairs done %.1fs"%(time.time()-t0))
# C(k)=P(both at sep k)/p0^2  ; baseline pairs = (NMAX-k)*p0^2
Ck=A[1:].astype(float)/((NMAX-np.arange(1,Kmax+1))*p0*p0)
# show a few structured k
print("\n  k    C(k)     note")
for k in [1,2,3,4,5,6,7,10,11,15,30,35,77,210,211]:
    note=[]
    for q in [5,7,11,13]:
        if k%q==0: note.append("%d|k"%q)
        if (3*k-1)%q==0: note.append("%d|3k-1"%q)
        if (3*k+1)%q==0: note.append("%d|3k+1"%q)
    print("%4d  %.4f   %s"%(k,Ck[k-1]," ".join(note)))

print("\nmean C(k) over k=1..%d : %.4f"%(Kmax,Ck.mean()))
print("sum (C(k)-1) k=1..%d   : %.4f"%(Kmax,(Ck-1).sum()))
import csv
with open(os.path.join(_DATA,"explore_fano.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["H","n_windows","mean","F_total"])
    for H,m,mu,var,F in fano: w.writerow([H,m,"%.6f"%mu,"%.6f"%F])
print("wrote explore_fano.csv  (exploratory pass, 6N<=2e8)")
print("TOTAL %.1fs"%(time.time()-t0))
