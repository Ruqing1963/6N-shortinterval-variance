import json, numpy as np, matplotlib
import os
_HERE=os.path.dirname(os.path.abspath(__file__))
_DATA=os.path.normpath(os.path.join(_HERE,"..","data"))
_FIG=os.path.normpath(os.path.join(_HERE,"..","figures"))
os.makedirs(_DATA,exist_ok=True); os.makedirs(_FIG,exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":0.3,
    "figure.dpi":150,"savefig.bbox":"tight"})

r=json.load(open(os.path.join(_DATA,"var_results.json"))); d=json.load(open(os.path.join(_DATA,"var_detrend.json")))
z=np.load(os.path.join(_DATA,"var_arrays.npz"))
p0=r["p0"]; G=r["G"]
kk=z["kk"]; Ck=z["Ck"]; C_HL=z["C_HL"]

det=d["detrend"]
Hd=np.array([x["H"] for x in det]); Ftot=np.array([x["F_total"] for x in det])
Floc=np.array([x["F_local"] for x in det]); Ftr=np.array([x["F_trend"] for x in det])

# HL-predicted Fano via exact variance identity, from C_HL
def F_pred(H,Cvec):
    k=np.arange(1,H); 
    if H-1>len(Cvec): return np.nan
    return (1-p0)+(2*p0/H)*np.sum((H-k)*(Cvec[:H-1]-1))
Hpred=[h for h in Hd if h<=3000]
Fhl=[F_pred(h,C_HL) for h in Hpred]

# ============ FIGURE 1 ============
fig,ax=plt.subplots(1,2,figsize=(9.4,4.0))
# (A) Fano decomposition
ax[0].axhline(1.0,color="0.5",lw=0.8,ls="--")
ax[0].semilogx(Hd,Ftot,"o-",ms=3,lw=1.2,color="#222222",label=r"$F_{\rm total}$ (all windows)")
ax[0].semilogx(Hd,Floc,"s-",ms=3,lw=1.2,color="#1f77b4",label=r"$F_{\rm local}$ (density-detrended)")
ax[0].semilogx(Hd,Ftr,"^-",ms=3,lw=1.2,color="#d62728",label=r"$F_{\rm trend}$ (density gradient)")
ax[0].axvline(410,color="0.7",lw=0.8,ls=":")
ax[0].annotate(r"$H^\ast\!\approx\!410$",(410,12),fontsize=7,rotation=90,va="center",ha="right",color="0.4")
ax[0].set_xlabel("window length $H$ (centres)"); ax[0].set_ylabel("Fano factor  Var/mean")
ax[0].set_title("(A) variance decomposition",fontsize=9)
ax[0].legend(fontsize=7,loc="upper left")
# (B) C(k) structure for small k
ax[1].axhline(1.0,color="0.6",lw=0.8,ls="--")
ax[1].axhline(G,color="#2ca02c",lw=0.8,ls=":")
ax[1].annotate(r"$G=%.3f$"%G,(150,G),fontsize=7,color="#2ca02c",va="bottom")
km=200
ax[1].plot(kk[:km],Ck[:km],"o",ms=2.5,color="#1f77b4",alpha=0.7,label="measured")
ax[1].plot(kk[:km],C_HL[:km],".",ms=2.0,color="#d62728",label="HL singular series")
for k,lab in [(5,"5|k"),(35,"5,7|k"),(70,"5,7|k")]:
    if k<=km: ax[1].annotate(lab,(k,Ck[k-1]),fontsize=6,xytext=(2,3),textcoords="offset points")
ax[1].set_xlabel("separation $k$ (centres)"); ax[1].set_ylabel(r"pair correlation $C(k)$")
ax[1].set_title("(B) local arithmetic: $C(k)$ vs Hardy--Littlewood",fontsize=9)
ax[1].legend(fontsize=7,loc="upper right")
fig.suptitle(r"Twin counts in short intervals ($6N\leq 6\times10^8$, $2.17\times10^6$ twin centres)",fontsize=10)
fig.savefig(os.path.join(_FIG,"p29_fig1.pdf")); print("fig1 done")

# ============ FIGURE 2 ============
fig2,ax2=plt.subplots(1,2,figsize=(9.4,4.0))
# (A) measured vs HL scatter, all k
lim=[0.3,3.5]
ax2[0].plot(lim,lim,"k-",lw=0.7,alpha=0.6,label="$y=x$")
ax2[0].scatter(C_HL,Ck,s=3,alpha=0.25,color="#1f77b4",edgecolors="none")
ax2[0].set_xlim(lim); ax2[0].set_ylim(lim)
ax2[0].set_xlabel(r"$C_{\rm HL}(k)$ prediction"); ax2[0].set_ylabel(r"$C(k)$ measured")
ax2[0].set_title(r"(A) $C(k)$: all $k\leq3000$ (median dev. 1.7%)",fontsize=9)
ax2[0].legend(fontsize=7,loc="upper left")
# (B) F_local vs HL-identity prediction (sub-Poisson regime)
ax2[1].axhline(1.0,color="0.6",lw=0.8,ls="--")
ax2[1].semilogx(Hd,Floc,"s-",ms=3,lw=1.2,color="#1f77b4",label=r"$F_{\rm local}$ measured (detrended)")
ax2[1].semilogx(Hpred,Fhl,"o--",ms=3,lw=1.0,color="#d62728",label=r"$F$ from HL $C(k)$ (identity)")
ax2[1].set_xlabel("window length $H$ (centres)"); ax2[1].set_ylabel("Fano factor  Var/mean")
ax2[1].set_title("(B) detrended variance follows the singular series",fontsize=9)
ax2[1].set_ylim(0.7,1.4)
ax2[1].legend(fontsize=7,loc="upper left")
fig2.savefig(os.path.join(_FIG,"p29_fig2.pdf")); print("fig2 done")
