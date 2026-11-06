"""Write the job queue for every Chemprop run of the study, in priority order.
Each line holds the arguments of one src/lib/train.py call. Usage: python3 -I make_jobs.py > jobs.txt"""
LEVELS = "within:0.02@{a},within:0.05@{a},within:0.1@{a},within:0.2@3,within:0.4@3,within:1.0@3"
CLEAN = "true," + LEVELS.format(a=10) + ",pair:centre@5,pair:spect@5,pair:mixed@5,cross@3,none"
RXM = ",rxnmapper,rxnmapper_raw,matched@10"
TRAINED = "true," + LEVELS.format(a=5)
B = {
    "rdb7": CLEAN + RXM,
    "e2sn2": CLEAN,
    "cyclo": "true," + LEVELS.format(a=5) + ",rxnmapper,matched@5",
}
jobs = []


def job(ds, s, train, tests, tag=None, extra=""):
    name = tag or train.replace(":", "")
    jobs.append(f"--data {ds} --split-seed {s} --train-cond {train} --test-conds {tests} --epochs 50 {extra}"
                f"--out results/grid/{ds}/{name}_s{s}")


ten, five, three = range(10), range(5), range(3)
# 1. clean models first (reproduction gate on the heavy atom setting)
for s in three:
    job("rdb7", s, "true", B["rdb7"])
    job("e2sn2", s, "true", B["e2sn2"])
# 2. gate: RDB7 explicit H against the Reply (4.9 +/- 0.2), split 0
jobs.append("--data rdb7 --split-seed 0 --train-cond explicit --test-conds explicit --epochs 50 --out results/repro/rdb7_explicit_s0")
# 3. E2/SN2 ten split core
for s in range(3, 10):
    job("e2sn2", s, "true", B["e2sn2"])
for s in ten:
    job("e2sn2", s, "within:0.1", TRAINED)
for s in five:
    job("e2sn2", s, "true", "true,within:0.1@10", tag="truerep", extra="--init-offset 100 ")
# 4. RDB7 five split core (cut from ten at 11:30 UTC: one RDB7 model takes about 30 min on the shared runner)
for s in range(3, 5):
    job("rdb7", s, "true", B["rdb7"])
for s in five:
    job("rdb7", s, "within:0.1", TRAINED + ",rxnmapper,matched@5")
    job("rdb7", s, "rxnmapper", "true,rxnmapper,matched@5,within:0.1@5")
for s in three:
    job("rdb7", s, "matched", "true,matched@5,rxnmapper,within:0.1@5")
    job("rdb7", s, "true", "true,within:0.1@10,rxnmapper,matched@10", tag="truerep", extra="--init-offset 100 ")
# 5. three split level grid and controls
for ds in ("e2sn2", "rdb7"):
    for s in three:
        for p in ("0.02", "0.05", "0.2", "0.4", "1.0"):
            job(ds, s, f"within:{p}", TRAINED)
        job(ds, s, "cross", TRAINED + ",cross@3")
        job(ds, s, "none", "none")
# 6. augmentation redrawn every epoch
for s in three:
    for p in ("0.1", "0.2"):
        job("e2sn2", s, f"aug:{p}", TRAINED)
for s in three:
    job("rdb7", s, "aug:0.1", TRAINED)
# 7. Cyclo-23-TS
for s in three:
    job("cyclo", s, "true", B["cyclo"])
    job("cyclo", s, "within:0.1", TRAINED + ",rxnmapper")
    job("cyclo", s, "rxnmapper", "true,rxnmapper,within:0.1@5")
    job("cyclo", s, "none", "none")
# 8. sensitivity checks and the remaining explicit H reproduction splits
job("rdb7", 0, "within:0.1", TRAINED, tag="within0.1_valclean", extra="--val-cond true ")
for s in (1, 2):
    jobs.append(f"--data rdb7 --split-seed {s} --train-cond explicit --test-conds explicit --epochs 50 --out results/repro/rdb7_explicit_s{s}")
for s in three:
    job("e2sn2", s, "true", "true,within:0.1@5", tag="true_ep100", extra="--epochs 100 ")
    job("e2sn2", s, "within:0.1", "true,within:0.1@5", tag="within0.1_ep100", extra="--epochs 100 ")
job("rdb7", 0, "true", "true,within:0.1@5", tag="true_ep100", extra="--epochs 100 ")
job("rdb7", 0, "within:0.1", "true,within:0.1@5", tag="within0.1_ep100", extra="--epochs 100 ")
# 9. added after the results audit (2026-11-04): unmapped E2/SN2 models on the remaining splits
# (bound on unmapped minus clean), and RDB7 levels 0.6 and 0.8 trained and tested, because the
# both regime crossover with the map free references fell between 0.4 and 1.0 (plan trigger)
for s in range(3, 10):
    job("e2sn2", s, "none", "none")
for s in three:
    for p in ("0.6", "0.8"):
        job("rdb7", s, f"within:{p}", f"true,within:{p}@3")
# 10. optional RDB7 headline splits 5 to 9
for s in range(5, 10):
    job("rdb7", s, "true", B["rdb7"])
    job("rdb7", s, "within:0.1", TRAINED + ",rxnmapper,matched@5")
    job("rdb7", s, "rxnmapper", "true,rxnmapper,matched@5,within:0.1@5")
# 11. added in the first revision (2026-11-04): the redrawn model evaluated on RXNMapper test maps
# (to test whether synthetic augmentation transfers to tool errors; same seeds as aug0.1,
# so the model is the same one), and four more Cyclo-23-TS clean models for the test only slope
# and the RXNMapper test gap, which three splits did not resolve
for s in three:
    job("rdb7", s, "aug:0.1", "true,within:0.1@3,rxnmapper", tag="aug0.1rx")
for s in range(3, 7):
    job("cyclo", s, "true", B["cyclo"])
# 12. added in the second revision (2026-11-04): the redrawn model on RXNMapper maps on the remaining
# RDB7 headline splits, so all three trainings are compared on eight splits; and clean models
# retrained with the checkpoint used for the profile condition (RXNMapper's per reaction error
# profiles on random reactions), which splits the concentration term into clustering and the
# choice of reactions. The clean retrain carries its own uniform curve so every term is paired.
for s in range(3, 8):
    job("rdb7", s, "aug:0.1", "true,within:0.1@3,rxnmapper", tag="aug0.1rx")
for s in range(8):
    job("rdb7", s, "true", "true,within:0.02@3,within:0.05@3,within:0.1@3,within:0.2@3,rxnmapper,matched@3,profile@3", tag="trueprof")
print("\n".join(jobs))
