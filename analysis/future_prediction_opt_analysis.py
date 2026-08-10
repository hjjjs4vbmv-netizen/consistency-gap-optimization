"""Optimizer-level future prediction: does early R_opt predict future FID?

Early R_opt (optimizer-state residual) at 32k/64k (tick1/2) from the retrained
future_pred_opt_0811 runs. Future FID at 256k from future_pred_0809 (same seed/
config). Leave-one-seed-out over 4 seeds.
"""
import numpy as np

# Early R_opt at tick1 (32k) and tick2 (64k), g=1.3 (where residual lives)
R_OPT_t1 = {  # g=1.3, tick1 (32k)
    0: 0.0184, 1: 0.0183, 2: 0.0203, 4: 0.0207,
}
R_OPT_t2 = {  # g=1.3, tick2 (64k)
    0: 0.0233, 1: 0.0209, 2: 0.0192, 4: 0.0228,
}
# Early R_opt at g=1.0 (reference)
R_OPT_t1_g10 = {0: 0.0138, 1: 0.0134, 2: 0.0131, 4: 0.0130}
R_OPT_t2_g10 = {0: 0.0168, 1: 0.0145, 2: 0.0172, 4: 0.0169}

# Future FID (256k, NFE1) from future_pred_0809
FID_13 = {0: 222.7, 1: 241.2, 2: 250.9, 4: 219.5}  # g=1.3
FID_10 = {0: 257.7, 1: 247.3, 2: 313.9, 4: 250.1}  # g=1.0
IMPROV = {s: FID_10[s]-FID_13[s] for s in [0,1,2,4]}  # g=1.0 - g=1.3

SEEDS = [0,1,2,4]

def pearson(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<3: return float('nan')
    xm,ym=x.mean(),y.mean()
    d=((x-xm)**2).sum()*((y-ym)**2).sum()
    return ((x-xm)*(y-ym)).sum()/np.sqrt(d) if d>0 else float('nan')

def spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
    return pearson(rx,ry)

print("="*70)
print("OPTIMIZER-LEVEL FUTURE PREDICTION (early R_opt -> future FID)")
print("="*70)

print("\n--- Data (g=1.3) ---")
print(f"{'seed':>4} {'R_opt_32k':>10} {'R_opt_64k':>10} {'FID_256k':>10} {'improv':>8}")
for s in SEEDS:
    print(f"{s:>4} {R_OPT_t1[s]:>10.4f} {R_OPT_t2[s]:>10.4f} {FID_13[s]:>10.1f} {IMPROV[s]:>8.1f}")

print("\n--- Cross-seed spread of early R_opt ---")
for label, R in [('R_opt_32k', R_OPT_t1), ('R_opt_64k', R_OPT_t2)]:
    v = np.array([R[s] for s in SEEDS])
    print(f"  {label}: mean={v.mean():.4f} std={v.std():.4f} range=[{v.min():.4f},{v.max():.4f}]")
# compare to R_grad spread
print(f"  (compare: R_grad_32k spread was 0.0085-0.0096, std ~0.0004)")

print("\n--- Q: Does early R_opt predict future FID (g=1.3)? ---")
for label, R in [('R_opt_32k', R_OPT_t1), ('R_opt_64k', R_OPT_t2)]:
    r = [R[s] for s in SEEDS]; fid = [FID_13[s] for s in SEEDS]; imp = [IMPROV[s] for s in SEEDS]
    print(f"  {label} vs FID(g=1.3):   pearson={pearson(r,fid):+.3f} spearman={spearman(r,fid):+.3f}")
    print(f"  {label} vs improvement: pearson={pearson(r,imp):+.3f} spearman={spearman(r,imp):+.3f}")

print("\n--- Q: Does early R_opt differ between g=1.0 and g=1.3 (is it gap-sensitive)? ---")
for s in SEEDS:
    d1 = R_OPT_t1[s] - R_OPT_t1_g10[s]
    print(f"  seed{s}: R_opt_32k diff(1.3-1.0)={d1:+.4f}")

print("\n" + "="*70)
print("VERDICT")
print("="*70)
print("""
Key observation: early R_opt at g=1.3 is nearly CONSTANT across seeds
(0.0183-0.0207 at 32k, a ~0.002 spread), just like R_grad was. The cross-seed
variation in future FID (219-251 at g=1.3) and improvement (6-63 FID) is far
larger than the predictor spread. A near-constant predictor cannot predict
strongly-varying outcomes.

If pearson(spearman) with FID/improvement is near-zero or unreliable (n=4),
the optimizer-level future-prediction thesis FAILS too.
""")
