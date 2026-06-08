import numpy as np, json, time
import os
_HERE=os.path.dirname(os.path.abspath(__file__))
_DATA=os.path.normpath(os.path.join(_HERE,"..","data"))
_FIG=os.path.normpath(os.path.join(_HERE,"..","figures"))
os.makedirs(_DATA,exist_ok=True); os.makedirs(_FIG,exist_ok=True)
t0=time.time()
LIMIT=600_000_000; NMAX=LIMIT//6      # 10^8 centres, matches Part XXVI scale
Kmax=3000

# ---------- prime sieve ----------
sieve=np.ones(LIMIT+3,dtype=bool); sieve[:2]=False
for p in range(2,int((LIMIT+2)**0.5)+1):
    if sieve[p]: sieve[p*p::p]=False
is_twin=np.zeros(NMAX+1,dtype=bool)
for a in range(1,NMAX+1,10_000_000):
    b=min(a+10_000_000,NMAX+1); idx=np.arange(a,b,dtype=np.int64)
    is_twin[a:b]=sieve[6*idx-1]&sieve[6*idx+1]
del sieve
ntwin=int(is_twin.sum()); p0=ntwin/NMAX
print("NMAX=%d 6N<=%d  ntwin=%d  p0=%.7f  (%.1fs)"%(NMAX,LIMIT,ntwin,p0,time.time()-t0))

# ---------- Fano factor F(H) over disjoint windows ----------
x=is_twin[1:NMAX+1].astype(np.int64)
cs=np.concatenate([[0],np.cumsum(x)]); del x
Hs=[5,7,10,15,20,30,50,70,100,150,200,300,400,500,700,1000,1500,2000,3000,
    5000,7000,10000,15000,20000,30000,50000,70000,100000]
fano=[]
print("\n   H        m       mean        var      Fano F     F-1")
for H in Hs:
    m=NMAX//H
    ends=np.arange(H,(m+1)*H,H)
    counts=cs[ends]-cs[ends-H]
    mu=float(counts.mean()); var=float(counts.var()); F=var/mu
    fano.append(dict(H=int(H),m=int(m),mean=mu,var=var,F=F))
    print("%7d %8d  %10.4f %11.4f  %.5f  %+.5f"%(H,m,mu,var,F,F-1))
del cs

# ---------- measured pair correlation C(k) ----------
pos=np.nonzero(is_twin[1:NMAX+1])[0]          # twin indices in 0..NMAX-1
pos=pos[pos<NMAX-Kmax-1]                       # keep those that can shift by Kmax
xb=is_twin[1:NMAX+1]
A=np.zeros(Kmax+1,dtype=np.int64)
for k in range(1,Kmax+1):
    A[k]=int(np.count_nonzero(xb[pos+k]))      # gather ~2M, fast
print("\npair counts done %.1fs"%(time.time()-t0))
npos=len(pos)
kk=np.arange(1,Kmax+1)
# baseline expected pairs at sep k = npos * p0  (each of npos anchors, prob p0 partner)
Ck=A[1:].astype(float)/(npos*p0)

# ---------- Hardy-Littlewood prediction C_HL(k) ----------
# smallest-prime-factor sieve to 3*Kmax+1
M=3*Kmax+2
spf=np.zeros(M+1,dtype=np.int64)
for i in range(2,M+1):
    if spf[i]==0:
        spf[i::i]=np.where(spf[i::i]==0,i,spf[i::i])
def distinct_primes_gt3(n):
    s=set()
    while n>1:
        p=spf[n]; 
        if p>3: s.add(p)
        while n%p==0: n//=p
    return s
# generic constant G = prod_{p>3}(p-4)p/(p-2)^2
# build a prime list to 2e6 with its own sieve
sp2=np.ones(2_000_001,dtype=bool); sp2[:2]=False
for p in range(2,1415):
    if sp2[p]: sp2[p*p::p]=False
primes=np.nonzero(sp2)[0]; primes=primes[primes>=5].astype(np.float64)
G=float(np.prod((primes-4)*primes/((primes-2)**2)))
print("generic 4-tuple correlation G = %.6f"%G)

C_HL=np.empty(Kmax)
for i,k in enumerate(kk):
    c=G
    for p in distinct_primes_gt3(int(k)):       # p|k  -> (p-2)/(p-4)
        c*=(p-2)/(p-4)
    for p in distinct_primes_gt3(3*int(k)-1):   # p|3k-1 -> (p-3)/(p-4)
        c*=(p-3)/(p-4)
    for p in distinct_primes_gt3(3*int(k)+1):   # p|3k+1 -> (p-3)/(p-4)
        c*=(p-3)/(p-4)
    C_HL[i]=c

resid=Ck/C_HL-1
print("\n  k    C_meas   C_HL    ratio-1")
for k in [1,2,3,5,7,11,35,77,210,211,1001,2310]:
    if k<=Kmax:
        print("%5d  %.4f  %.4f  %+.4f"%(k,Ck[k-1],C_HL[k-1],resid[k-1]))
print("\nmedian |C_meas/C_HL - 1| = %.4f"%np.median(np.abs(resid)))
print("mean C_meas = %.4f   mean C_HL = %.4f"%(Ck.mean(),C_HL.mean()))
print("sum(C_meas-1) k<=%d = %.3f   sum(C_HL-1) = %.3f"%(Kmax,(Ck-1).sum(),(C_HL-1).sum()))

# ---------- variance identity cross-check ----------
# F(H) = (1-p0) + (2 p0 / H) * sum_{k=1}^{H-1} (H-k)(C(k)-1)
def F_pred(H,Cvec):
    kk2=np.arange(1,H); 
    return (1-p0)+(2*p0/H)*np.sum((H-kk2)*(Cvec[:H-1]-1))
print("\n  H    F_meas   F_pred(C_meas)  F_pred(C_HL)")
checkH=[100,300,500,1000,2000,3000]
idcheck=[]
for H in checkH:
    Fm=[f["F"] for f in fano if f["H"]==H][0]
    fp_meas=F_pred(H,Ck); fp_hl=F_pred(H,C_HL)
    idcheck.append(dict(H=H,F_meas=Fm,F_pred_meas=float(fp_meas),F_pred_HL=float(fp_hl)))
    print("%5d  %.4f    %.4f         %.4f"%(H,Fm,fp_meas,fp_hl))

import csv
def Hstar_crossover(fano):
    f=sorted(fano,key=lambda r:r["H"])
    for i in range(1,len(f)):
        if f[i]["F"]>=1>f[i-1]["F"]:
            (H1,F1),(H2,F2)=(f[i-1]["H"],f[i-1]["F"]),(f[i]["H"],f[i]["F"])
            return float(np.exp(np.log(H1)+(1-F1)*(np.log(H2)-np.log(H1))/(F2-F1)))
    return float("nan")
Hstar=Hstar_crossover(fano)

# pair_correlation.csv : k, measured C(k), Hardy-Littlewood prediction
with open(os.path.join(_DATA,"pair_correlation.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["k","C_measured","C_HL_prediction"])
    for k,cm,ch in zip(kk,Ck,C_HL): w.writerow([int(k),"%.6f"%cm,"%.6f"%ch])
# fano_ladder.csv : H, n_windows, mean, var, F_total
with open(os.path.join(_DATA,"fano_ladder.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["H","n_windows","mean","var","F_total"])
    for row in fano: w.writerow([row["H"],row["m"],"%.6f"%row["mean"],"%.6f"%row["var"],"%.6f"%row["F"]])
# variance_identity.csv : H, F_measured, F from measured C, F from HL C
with open(os.path.join(_DATA,"variance_identity.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["H","F_measured","F_from_C_measured","F_from_C_HL"])
    for ic in idcheck: w.writerow([ic["H"],"%.6f"%ic["F_meas"],"%.6f"%ic["F_pred_meas"],"%.6f"%ic["F_pred_HL"]])
# summary.csv : scalar parameters and aggregate results
with open(os.path.join(_DATA,"summary.csv"),"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["parameter","value"])
    for k,v in [("LIMIT_6N_max",LIMIT),("NMAX_centres",NMAX),("n_twin_centres",ntwin),
        ("p0_global_twin_rate","%.7f"%p0),("G_generic_4tuple_correlation","%.6f"%G),
        ("Kmax_pair_separation",Kmax),("crossover_H_star","%.1f"%Hstar),
        ("sum_Cmeas_minus_1_kle%d"%Kmax,"%.3f"%float((Ck-1).sum())),
        ("sum_CHL_minus_1_kle%d"%Kmax,"%.3f"%float((C_HL-1).sum())),
        ("mean_C_measured","%.4f"%float(Ck.mean())),("mean_C_HL","%.4f"%float(C_HL.mean())),
        ("median_abs_rel_resid_C","%.4f"%float(np.median(np.abs(resid))))]:
        w.writerow([k,v])
print("\nCSVs written to %s ; crossover H* = %.1f"%(_DATA,Hstar))
print("TOTAL %.1fs"%(time.time()-t0))
