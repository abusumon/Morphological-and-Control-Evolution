"""Generate the README figures from results/ga_log.csv and results/validation.csv."""
import csv
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("results/ga_log.csv")))
gen = [int(r["generation"]) for r in rows]
best = [float(r["best_so_far"]) for r in rows]
mean = [float(r["gen_mean"]) for r in rows]
med = [float(r["gen_median"]) for r in rows]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(gen, best, label="best-so-far", lw=2)
ax.plot(gen, mean, label="generation mean", alpha=0.7)
ax.plot(gen, med, label="generation median", alpha=0.7)
ax.axvline(87, color="k", ls="--", lw=1)
ax.annotate("best genome found (gen 87)\nunchanged through gen 277",
            xy=(87, 126.5), xytext=(120, 60),
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("generation"); ax.set_ylabel("fitness")
ax.set_title("Stage 2 fitness vs generation (6 training spawns)")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("docs/assets/fitness_curve.png", dpi=150)

vrows = list(csv.DictReader(open("results/validation.csv")))
vg = [int(r["generation"]) for r in vrows]
vt = [float(r["train_fitness"]) for r in vrows]
vv = [float(r["val_fitness"]) for r in vrows]
vr = [f"{r['val_reached']}/{r['val_total']}" for r in vrows]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(vg, vt, "o-", label="training fitness (best-so-far)")
ax.plot(vg, vv, "s-", label="held-out validation fitness")
for x, t, v, r in zip(vg, vt, vv, vr):
    ax.annotate(f"{r} reached", xy=(x, v), xytext=(0, -18),
                textcoords="offset points", ha="center", fontsize=8)
ax.annotate("validation peak", xy=(49, 104.97), xytext=(60, 110),
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlabel("generation"); ax.set_ylabel("fitness")
ax.set_title("Training vs held-out validation fitness — validation peaks at gen 49,\n"
             "then declines while training fitness keeps climbing (mild overfitting)")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig("docs/assets/train_vs_validation.png", dpi=150)
print("wrote docs/assets/fitness_curve.png and docs/assets/train_vs_validation.png")
