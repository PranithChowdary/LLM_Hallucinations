# Research Workbook
**How this works:** this document tells you *what the project needs*, not who does what. You four split the work among yourselves, renegotiate the split when it stops making sense, and everybody ends up understanding the whole pipeline, not just their corner of it.

Standing rule on AI assistance: use it freely, but you must be able to explain every line to someone trying to poke holes in it, and teach it back in when someone asks. Log everything properly just to understand everything better even after something goes wrong 3-4 weeks later.

---

## Part 1 - What we're building

A two-stage system that decides whether an LLM's answer can be trusted:

1. A **cheap single-pass filter** reads signals from one generation (token log-probabilities, entropy, linguistic features) and tries to certify the answer as safe.
2. Anything it can't certify **escalates** to an expensive multi-sample detector (SelfCheckGPT / Semantic Entropy).
3. Both stages are **conformally calibrated**, so the system carries a formal, distribution-free guarantee on recall — not a threshold someone eyeballed.

The research question: *how much of that guarantee survives when the calibrated score comes from a single generation instead of many, and how much inference cost does the cascade save?*

---

## Part 2 - The full experiment list

Everything the paper needs. Roughly dependency-ordered - later experiments need earlier ones done.

### Block A - Data & generation
- [ ] **A1.** Load HaluEval (QA, dialogue, summarization) and TruthfulQA; normalize both into one schema. Understand the label semantics of each — what exactly does "hallucination" mean in each dataset, and are those definitions compatible? Write down the answer; it goes in the paper.
- [ ] **A2.** Generate answers with the chosen open model over both datasets. Fixed seeds, logged configs, cached to disk so nobody regenerates unnecessarily.
- [ ] **A3.** Capture per-token logits and hidden states during generation via forward hooks.
- [ ] **A4.** Sanity-check the generation set: label balance, length distributions, obvious degenerate outputs. Catch data problems here, not in month three.

### Block B - Signals
- [ ] **B1.** Extract Tier-1 single-pass features: mean/min token log-probability, log-probability variance, token entropy, fraction of low-confidence tokens, output length.
- [ ] **B2.** Extract Tier-2 linguistic features: hedging frequency, entity/numeral density, assertion density, repetition.
- [ ] **B3.** Reproduce **SelfCheckGPT** from the paper.
- [ ] **B4.** Reproduce **Semantic Entropy** from the paper (semantic clustering via NLI, then entropy over clusters).
- [ ] **B5.** Reproduce **Semantic Entropy Probes** — the cheap linear-probe alternative, and the closest existing prior art to our own approach.
- [ ] **B6.** Naive log-probability threshold baseline — the weakest baseline, but the paper needs it.
- [ ] **B7.** Feature correlation analysis: which of our signals are redundant with each other?
- [ ] **B8.** Train the combined single-pass classifier (logistic regression + gradient-boosted trees) on B1+B2 features.
- [ ] **B9.** Sampling budget sweep: run Semantic Entropy at N = 1, 2, 4, 8, 16. Where does it saturate?

### Block C - Calibration
- [ ] **C1.** Implement split-conformal risk control.
- [ ] **C2.** Synthetic test harness: scores with a known, controllable relationship to labels; assert empirical coverage converges to target as calibration size grows. **This must pass before any calibration touches real data.**
- [ ] **C3.** Calibration/test splitting utilities — stratified, seeded, with explicit leakage checks (no overlap, no near-duplicate questions across splits).
- [ ] **C4.** Calibrate the sampling-based score (reproducing the known conformal-abstention approach).
- [ ] **C5.** Calibrate the single-pass score (our new comparison point).
- [ ] **C6.** Coverage validation: repeat calibration/test splits many times, check the guarantee empirically holds for both C4 and C5.
- [ ] **C7.** Calibration set size sweep: how small before the guarantee gets uselessly conservative?
- [ ] **C8.** Exchangeability stress test: calibrate on one domain, test on another; document how coverage degrades. This becomes the limitations section.

### Block D - Cascade
- [ ] **D1.** Build the two-stage router: calibrated single-pass filter → escalate uncertified cases to the calibrated sampling detector.
- [ ] **D2.** Cost accounting: generations, tokens, wall-clock per method. Pick the cost unit early and defend it in the paper.
- [ ] **D3.** Cost-reduction-at-fixed-recall — the headline experiment.
- [ ] **D4.** Pareto frontier: detection quality vs. cost, every method on one figure.
- [ ] **D5.** Risk–coverage curves: remaining hallucination rate if the system abstains on the riskiest K%.
- [ ] **D6.** ECE / calibration curves for the final cascade.

### Block E - Analysis
- [ ] **E1.** Confidence × correctness quadrants — specifically, how often does the cheap filter certify a *confident hallucination* as safe? This is the dangerous case.
- [ ] **E2.** Disagreement analysis: dump every case where the cheap filter and the expensive detector disagree, read them by hand, categorize.
- [ ] **E3.** Per-domain breakdown: does anything change across HaluEval's QA vs. dialogue vs. summarization subsets?
- [ ] **E4.** Model generalization (stretch): does the story hold on a second model family?

### Block F - Write-up
- [ ] **F1.** Literature check, repeated close to submission — this area moves fast, and a check done in September doesn't hold in December.
- [ ] **F2.** Figures and tables, final versions.
- [ ] **F3.** Draft: method, results, limitations, related work.
- [ ] **F4.** Reviewer-#2 pass and revision.

---

## Part 3 - What the final repo should look like

```
hallucination-cascade/
├── README.md                      # a stranger reproduces the paper from this
├── pyproject.toml                 # pinned deps
├── .pre-commit-config.yaml        # ruff + black
├── .github/workflows/ci.yml       # ruff + pytest on every PR
│
├── configs/                       # one file per experiment — no hardcoded params in scripts
│   ├── generation/
│   ├── calibration/
│   └── cascade/
│
├── src/
│   ├── data/
│   │   ├── loaders.py             # HaluEval, TruthfulQA → unified schema
│   │   └── splits.py              # stratified, seeded, leakage-checked
│   ├── generation/
│   │   ├── generate.py
│   │   └── hooks.py               # logit + hidden-state capture
│   ├── signals/
│   │   ├── single_pass.py         # Tier 1 + Tier 2 features
│   │   ├── selfcheck.py
│   │   ├── semantic_entropy.py
│   │   └── probes.py
│   ├── calibration/
│   │   └── conformal.py           # split-conformal risk control
│   ├── cascade/
│   │   ├── router.py
│   │   └── costs.py
│   ├── eval/
│   │   ├── metrics.py             # AUROC, AUPRC, bootstrap CIs, ECE
│   │   ├── coverage.py            # guarantee validation
│   │   └── plots.py               # Pareto, risk-coverage, reliability
│   └── utils/
│       ├── wandb_logging.py       # one wrapper everyone uses
│       └── seeding.py
│
├── experiments/                   # thin scripts: load config, call src/, log to W&B
│   ├── run_generation.py
│   ├── run_signals.py
│   ├── run_calibration.py
│   └── run_cascade.py
│
├── tests/
│   ├── test_conformal.py          # the synthetic coverage test — non-negotiable
│   ├── test_splits.py             # leakage assertions
│   └── test_metrics.py
│
├── learning/                      # your Phase 0 notebooks
├── notebooks/                     # exploration only — never the source of a reported number
└── paper/                         # LaTeX, figures
```

Two rules about this structure: **`experiments/` scripts stay thin** (load config, call `src/`, log results) so logic lives in one testable place; and **nothing in `notebooks/` ever produces a number that appears in the paper** — if an exploration turns into a result, it moves into `src/` with a test first.

---

**W&B logging** - a run that isn't logged didn't happen.

Project: `hallucination-cascade`. Run naming: `{block}-{experiment}-{variant}`, e.g. `B4-semantic-entropy-n8`, `C6-coverage-target95`.

Always in config:
```
model_name, model_revision, dataset, dataset_split, n_examples,
seed, temperature, n_samples, detector_method, target_recall,
calibration_size, git_commit_sha
```
`git_commit_sha` matters more than it looks - it's what makes a run traceable to the exact code that produced it.

Always as metrics:
```
auroc, auprc, achieved_recall, target_recall, coverage_gap,
escalation_rate, cost_generations, cost_wallclock_s, ece
```

Artifacts: generated answers, extracted features, and calibration/test split indices. Tables: per-example predictions with scores and labels — this is what makes reproducibility possible without re-running anything.


**Research notebook** - each of you keeps one (digital or hand writen as per your comfort). What you tried, what happened, what you concluded, what you'd do differently. This habit does more for your development as researchers than any single technical skill on this list.
