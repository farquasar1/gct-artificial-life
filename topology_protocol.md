# Topology experiment: fixed protocol, round 3

Written before the first integration, 7 November 2026, 18:30 Europe/Lisbon.
No external preregistration is claimed. This is a prospective local analysis plan.

Prediction: the fraction of required intermediate producers lying in a supercritical
maintenance SCC, estimated by independent near-washout concentration interventions,
will improve held-out prediction of construction success beyond matched starting
stocks/initial flux and subsequently measured aggregate throughput. Test by decrease
in seed-grouped cross-validated log loss, with paired seed-bootstrap 95% interval.
An interval containing zero counts as inconclusive; an increase counts against it.
This is a constructed test of a narrow mechanism, not general validation of GCT.

Six catalysts, two directed 3-cycles, one incoming/outgoing edge per node. Source
rates: (.28,.28,.28,.15,.15,.15), multiplied by six paired lognormal factors, CV .12.
Topology A: cycles (0,1,2),(3,4,5). Topology B: (0,1,3),(2,4,5).
Same rate attached to each source, same catalyst stocks .015 each, F=1, I=0;
therefore identical initial total regeneration and intermediate production flux.
Three intermediates produced by nodes 0,2,3; beta=.04, mu=.1; shared food D=1,
decay d=.2, identical food inflow, initial resource stock and budget.
No topology destroys all cycles; edges, degrees and rate multiset are preserved.
Both receive the same 90% catalyst washout (.15 -> .015) at time zero.
All catalysts remain inside the declared six-node system; candidate cores need not
contain every node. Each SCC is tested at F=1, and competition may invalidate its
local growth prediction. Boundary support is only declared food, no immigration.

300 seeds, master 20261108. Integrate to 170, assay at 150 for 20 time units.
Four separate held-out substrate branches, absent before assay: intermediate pairs
(0,1),(0,2),(1,2), triple (0,1,2). All substrates start at 1, P=0.
dP/dt = q S product(I_required), q=100 for pairs, 5000 for triple.
These fixed q give equal hazard .04 at I=.02 for each required intermediate.
Success P>=.5. Record candidate-core minimum concentration >=.001 throughout assay
separately; empty candidate core is not viable. Common input budget=172 (initial
food 1, inflow 170, assay substrate 1), excluding identical starting catalysts.
Exact exposure identity remains a downstream oracle, not a new GCT prediction.

Independent infinitesimal concentration intervention on RHS at F=1,c=1e-7,
epsilon=1e-8 estimates the instantaneous off-diagonal production Jacobian; edge
threshold 1e-6, local supercriticality rho>.2. Compare estimated vs known matrices.

Baseline logistic ridge (penalty 1, intercept unpenalized), 5 folds grouped by seed
(seed modulo 5): product fixed effects, initial total catalyst stock, initial total
regeneration and intermediate flux; log1p cumulative regeneration flux, total
intermediate-production flux and total catalyst exposure, all integrated over 0-170;
include squared and cross terms of these 3 throughput measures. Augmented model
adds required-producer core fraction and all-required-in-core indicator. Scaling
uses training folds only. Report log loss, Brier score, AUC, paired seed-bootstrap
10,000-resample CI (seed 8813) for baseline minus augmented loss. These are
out-of-seed, not out-of-topology/construction tests. Post-treatment adjustment is
predictive, not an identified causal direct effect. Paired yield differences by
construction get seed-bootstrap intervals. Report subsequent throughput mismatch.
Do not change parameters/cutoffs if results disappoint. No Gillespie in this round.
