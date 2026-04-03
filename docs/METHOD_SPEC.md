# METHOD_SPEC

This file is the exact method and evaluation object for the locked paper.

## Problem setup
We study **top-down causal abstractions** of a fixed low-level model \(M_s\) on a fixed task setting \(s\).

The paper evaluates **candidate abstractions**, not automatic discovery over unconstrained hypothesis spaces.

## Requirements index
- **R1** candidate abstraction representation
- **R2** site universe
- **R3** high-level model library
- **R4** map families and hypergrids
- **R5** intervention dataset schema
- **R6** held-out and shift split logic
- **R7** structural description length
- **R8** parameter-count rules across map classes
- **R9** residual description length
- **R10** total bits and Pareto view
- **R11** null/control pool
- **R12** complexity and tuning-budget matching
- **R13** null-frontier estimation
- **R14** acceptance rule
- **R15** equivalence-class handling
- **R16** robustness code variant
- **R17** minimum viable empirical package
- **R18** positive-framing invalidation criteria

## R1. Candidate abstraction representation

### Intuition
A candidate abstraction must explicitly say:
1. which high-level algorithm is proposed,
2. which low-level sites instantiate each internal variable,
3. how each variable is read from activations,
4. how variable-level interventions are realized.

### Precise mechanism
A candidate abstraction is

\[
A = (H, \mathcal{V}_{int}, b, \{S_v\}_{v \in \mathcal{V}_{int}}, \{\tau_v\}_{v \in \mathcal{V}_{int}}, m, h)
\]

where:
- \(H\): a symbolic high-level model from the pre-registered library for setting \(s\),
- \(\mathcal{V}_{int} = \{N1, N2, R\}\): the non-output variables,
- \(b \in \{1,2,4\}\): per-variable site budget,
- \(S_v \subseteq \mathcal{U}_s\): the low-level site subset assigned to variable \(v\),
- \(|S_v| = b\) for all \(v\),
- the \(S_v\) are **disjoint**,
- \(\tau_v\): fitted readout map for \(v\),
- \(m\): map family,
- \(h\): family hyperparameters.

Low-level interventions are fixed by the partition:
- to intervene on variable \(v\), patch the residual activations at sites \(S_v\) from a source run into a base run.

No learned low-level intervention translator is used in v1.

### Implementation hooks
- store sites as integer pairs `(layer_index, token_index)`,
- store `S_v` as disjoint ordered lists,
- implement one readout module per variable,
- compute abstract states before and after variable-swap patching,
- do not allow overlapping site groups in v1.

## R2. Site universe

### Intuition
The site universe must be simple, explicit, and comparable across settings.

### Precise mechanism
The site universe is the set of **atomic residual-stream sites**:

\[
\mathcal{U}_s = \{(\ell, t) : \ell \in \{0,\dots,L_s\},\ t \in \{0,\dots,T_s-1\}\}
\]

where:
- \(\ell = 0\) denotes the post-embedding residual stream,
- \(\ell > 0\) denotes the post-block residual stream after transformer block \(\ell\),
- \(T_s\) is the fixed prompt length for setting \(s\).

This choice is locked for v1. No neurons, SAE features, attention edges, or overlapping subspaces.

### Implementation hooks
- prompt families must be validated to have fixed token length,
- save a site-index table per setting,
- candidate selection and structural coding operate only on this atomic site set.

## R3. High-level model library

### Intuition
The paper is top-down. The high-level algorithms are small, hand-written, and pre-registered.

### Precise mechanism
All three settings use the same abstract schema:
- \(N1\): identity of the first name/token,
- \(N2\): identity of the second name/token,
- \(R \in \{1,2\}\): which slot is repeated as the subject,
- \(Y\): output token identity.

Candidate library \(\mathcal{H}\) has four models:
1. **H_true_other**: \(Y = N2\) if \(R=1\), else \(Y = N1\).
2. **H_first**: \(Y = N1\).
3. **H_second**: \(Y = N2\).
4. **H_rep**: \(Y = N1\) if \(R=1\), else \(Y = N2\).

The library is finite and pre-registered. In v1 all four get equal structural code length.

### Implementation hooks
- implement `H` as small deterministic Python objects or symbolic functions,
- output domain is the name/symbol vocabulary of the setting,
- use the same intervention types \(\{swap(N1), swap(N2), swap(R)\}\) across all `H`.

## R4. Map families and hypergrids

### Intuition
Map classes must span both simple and expressive readouts, but stay small enough to compare cleanly.

### Precise mechanism
Primary map families:
1. `linear_dense`
2. `linear_sparse_l1`
3. `linear_lowrank`
4. `mlp1_relu`

Readout family is shared across variables; variable-specific output heads differ only by class count.

Class counts are locked:
- `c_N1 = |\mathcal{N}_s|`
- `c_N2 = |\mathcal{N}_s|`
- `c_R = 2`

The S2/S3 `OTHER` output-closure rule affects only observed output tokens and residual coding. It does **not** change the latent readout class sets for `N1` or `N2`.

Readout semantics are locked:
- each `tau_v` outputs class logits,
- training uses cross-entropy on the known latent label of variable `v`,
- the abstract state used by the symbolic model is the discrete class prediction
  \[
  z_v(x;A) = \arg\max \tau_v(a_{S_v}(x)),
  \]
- full-candidate residual bits are computed from the symbolic output induced by these discrete states,
- do **not** execute \(H\) on soft probabilities or logits in v1.

Hypergrids:
- `linear_dense`: no family-specific hyperparameter
- `linear_sparse_l1`: \(\lambda \in \{1e{-4}, 1e{-3}, 1e{-2}\}\)
- `linear_lowrank`: rank \(r \in \{4, 8, 16\}\)
- `mlp1_relu`: hidden width \(w \in \{32, 64, 128\}\)

The same family and hyperparameter choice is used for all three variables inside one candidate abstraction.

### Implementation hooks
- all inputs should be standardized using train-set statistics,
- all families must produce logits over the locked class set,
- variable-level prescreening uses validation cross-entropy,
- same training loop structure across families,
- do not add deeper MLPs or transformers in v1.

### Locked candidate-generation heuristic
Candidate generation is **not** part of the paper’s novelty, so it is deliberately simple and fixed.

For each setting, high-level model \(H\), map family \(m\), hyperparameter \(h\), and site budget \(b\):

1. **Single-site prescreen**
   - For each variable \(v \in \{N1,N2,R\}\) and each site \(u \in \mathcal U_s\), fit a single-site readout for \(v\) on train and score variable-level validation NLL.
   - Keep the top **q = 8** sites per variable.

2. **Variable-level group proposals**
   - For each variable, seed from the top **r = 4** prescreened single sites.
   - For \(b>1\), greedily add sites from that variable’s top-8 pool to maximize variable-level validation NLL reduction until \(|S_v|=b\).
   - This yields up to 4 proposed site groups per variable.

3. **Full candidate proposals**
   - Form the Cartesian product of the 4 proposed groups for `N1`, `N2`, and `R`, giving at most **64** raw full proposals.
   - Filter out any raw full proposal whose site groups are not pairwise disjoint.
   - Score only the remaining valid full proposals.

4. **Full-candidate scoring**
   - Fit all three readouts for each valid full proposal and score the full abstraction by validation residual bits.
   - Keep the best valid proposal.

5. **One-pass local refinement**
   - Starting from the best proposal, try single-site swaps within each variable from that variable’s top-8 pool.
   - Every proposed swap must preserve pairwise disjointness across `S_N1`, `S_N2`, and `S_R`.
   - Skip invalid swaps.
   - Accept the best improvement in validation residual bits.
   - Stop after one full sweep with no improvement.

The final refined proposal is the candidate for that `(H, m, h, b)` tuple.

Fallback rule:
- If disjointness filtering leaves `1..63` valid full proposals, score all valid proposals and proceed.
- If disjointness filtering leaves `0` valid full proposals, mark that `(H,m,h,b)` cell as `unevaluable_due_to_disjointness`.
- Do not backfill from lower-ranked groups.
- Do not regenerate proposals.
- Do not widen the search budget.
- Apply the same rule to ordinary candidate cells and `shuffled_pair` null cells.

### Implementation hooks
- prescreen uses variable-level NLL, not full abstraction score,
- full proposal scoring uses full abstraction residual bits under discrete `argmax` readouts,
- save all 64 proposal summaries where feasible,
- do not add beam search or larger search spaces in v1.

## R5. Intervention dataset schema

### Intuition
Held-out intervention evaluation is the paper’s evidentiary core.

### Precise mechanism
There are **two** data objects. Do not collapse them.

1. **Base/source tuple dataset** (candidate-independent)
\[
t_i = (x_b, x_s, v, meta_i)
\]
where:
- \(x_b\): base input,
- \(x_s\): source input,
- \(v \in \{N1, N2, R\}\): abstract variable being swapped,
- `meta_i`: latent assignment, template family, names, prompt identifiers, and a stable group ID.

2. **Candidate-scored intervention record** (candidate-dependent)
\[
d_i(A) = (t_i, y_i(A), \hat y_i(A))
\]
where:
- \(y_i(A)\): low-level model output token obtained by applying candidate \(A\)'s sitewise residual patch on the sites \(S_v\) associated with intervention type \(v\),
- \(\hat y_i(A)\): symbolic output token predicted by candidate \(A\) after reading abstract states, swapping variable \(v\), and executing \(H\).

For every base/source tuple:
- base and source differ only in the abstract variable \(v\),
- split assignment is defined on the tuple \(t_i\), not on \(d_i(A)\),
- the observed output \(y_i(A)\) is candidate-specific because the low-level intervention sites are candidate-specific.

This means:
- never precompute one global label \(y_i\) for all candidates,
- cache base/source activations if useful,
- compute candidate-specific patched outputs during candidate scoring.

### Implementation hooks
- generate tuple datasets from latent tuples first, then render prompts,
- keep all rows derived from one base-source tuple grouped together for splitting and bootstrapping,
- store actual low-level output tokens, not just correctness flags,
- save per-group residual-bit contributions for every scored candidate and every scored null candidate.

### Output-vocabulary closure

Locked rule:
- S1 uses the symbol vocabulary only, so `K_s = |\mathcal{N}_s|` and there is no `OTHER` class.
- S2 and S3 use a closed observed output alphabet `\mathcal{N}_s \cup {OTHER}`.
- In S2/S3, if the patched low-level model's argmax next token is not in the locked name vocabulary `\mathcal{N}_s`, record `y_A(t) = OTHER`.
- Do not drop such tuples.
- Do not project or alter the symbolic prediction `\hat y_A(t)`. Symbolic outputs remain in `\mathcal{N}_s` only.

Therefore:
- `K_s = |\mathcal{N}_s|` for S1
- `K_s = |\mathcal{N}_s| + 1` for S2/S3

### Locked S1 construction

S1 is a direct symbolic/state setting. It does not use natural-language prompts.

Locked S1 constants:
- `|\mathcal{N}_s| = 8`
- `R \in {1,2}`
- `family \in {canonical, shift}`
- positions:
  `0:N1_in, 1:N2_in, 2:R_in, 3:F_in, 4:relay_N1, 5:relay_N2, 6:relay_R, 7:nuis, 8:query`
- residual width `d_model = 32`
- `3` residual blocks after embeddings
- `s1_plant_seed = 0`

The planted true abstraction in S1 is:
- `H = H_true_other`
- `b = 2`
- `S*_N1 = {(1,4),(2,4)}`
- `S*_N2 = {(1,5),(2,5)}`
- `S*_R  = {(1,6),(2,6)}`
- `m = linear_dense`
- `h = default_dense`

Construction rule:
- relay positions `4,5,6` carry family-invariant partial codes for `N1`, `N2`, and `R`
- layer-1 and layer-2 relay codes are complementary
- nuisance position `7` carries family-dependent mixed codes
- the output token is produced from the decoded relay states under `H_true_other` plus a small nuisance bias from position `7`

The implementation must make each single relay site intentionally insufficient for exact recovery, while the paired true sites are exactly decodable. This is a planted-recovery requirement, not a tuning choice.

### Fixed prompt families for S2 and S3
Default canonical family:
- `"When {N1} and {N2} met, {SUBJ} gave a gift to"`

Default shift family:
- `"After {N1} and {N2} spoke, {SUBJ} handed a note to"`

`SUBJ` is `N1` if `R=1`, else `N2`.

These exact lexical defaults are locked. The only permitted change is lexical replacement needed to satisfy fixed-length tokenizer constraints in GPT-2-small; if that happens, record it as an empirical gate resolution rather than a silent method change.

## R6. Held-out and shift split logic

### Intuition
High-capacity fits must not be allowed to memorize intervention tuples or prompt templates.

### Precise mechanism
Within each setting:
- split at the **base-source tuple** level, not the individual intervention row level,
- train / val / test = 60 / 20 / 20 within the canonical template family,
- shift split = a separate prompt family with no training examples and no refitting.

Grouping key:
- latent variable assignment for \(N1,N2,R\),
- prompt template ID,
- source/base pair identity.

For name-based settings:
- enforce disjoint base-source tuple groups across train/val/test,
- keep the same name vocabulary across canonical and shift families in v1.

### Implementation hooks
- implement a deterministic split function from latent tuple hashes,
- save split manifests,
- do not mix shift-family tuples into train or validation.

## R7. Structural description length

### Intuition
The explanatory object includes the algorithm choice, site choices, family choice, hyperparameter choice, and fitted readout complexity.

### Precise mechanism
Primary structural code length is

\[
L_{struct}(A) = L(H) + L(b) + L(S\_{N1},S\_{N2},S\_R) + L(m) + L(h) + L_{\theta}(A)
\]

with:

1. **High-level model code**
\[
L(H) = \log_2 |\mathcal{H}| = 2 \text{ bits}
\]

2. **Per-variable site budget code**
\[
L(b) = \log_2 3
\]
because \(b \in \{1,2,4\}\).

3. **Disjoint site-set code**
Sequential combinatorial code in fixed variable order `(N1, N2, R)`:

\[
L(S\_{N1},S\_{N2},S\_R)
= \log_2 {|\mathcal{U}| \choose b}
+ \log_2 {|\mathcal{U}|-b \choose b}
+ \log_2 {|\mathcal{U}|-2b \choose b}
\]

4. **Map-family code**
\[
L(m) = \log_2 4 = 2 \text{ bits}
\]

5. **Hyperparameter code**
\[
L(h) = \log_2 |\mathcal{G}_m|
\]
where \(\mathcal{G}_m\) is the family-specific hypergrid.

6. **Parameter code**
Primary choice:
\[
L_{\theta}(A) = \frac{p_{eff}(A)}{2}\log_2 n_{train}
\]

where \(p_{eff}(A)\) is the effective number of fitted free parameters across all readout maps.

Constant candidate-independent terms are omitted from comparisons.

### Implementation hooks
- compute `L_sites` exactly from the site universe size and the chosen disjoint sets,
- use the same `n_train` for all candidate comparisons within a setting,
- do not substitute ad hoc L1 penalties for `L_theta`.

## R8. Parameter-count rules across map classes

### Intuition
Cross-family comparisons must use a single explicit counting rule.

### Precise mechanism
Let \(d_v = b \cdot d_{model}\) be the input dimensionality for variable \(v\), and \(c_v\) its output class count.

Locked class counts:
- \(c_{N1}=|\mathcal{N}_s|\)
- \(c_{N2}=|\mathcal{N}_s|\)
- \(c_R=2\)

Per-variable effective parameter counts:
- `linear_dense`: \(p_v = d_v c_v + c_v\)
- `linear_sparse_l1`: \(p_v = nnz(W_v) + c_v\), where `nnz` uses threshold \(10^{-8}\)
- `linear_lowrank`: \(p_v = r(d_v + c_v) + c_v\)
- `mlp1_relu`: \(p_v = d_v w + w + w c_v + c_v\)

Then:
\[
p_{eff}(A) = \sum_{v \in \{N1,N2,R\}} p_v
\]

### Implementation hooks
- compute `p_eff` after fitting,
- for sparse models, count thresholded nonzeros,
- do not use raw dense parameter count for sparse models,
- do not count optimizer state or activation caches as model parameters.

## R9. Residual description length

### Intuition
Residual code length measures how well the candidate compresses **held-out interventional behavior**.

### Precise mechanism
For each candidate abstraction and each scored intervention record \(d_i(A)\), let \(\hat y_i(A)\) be the symbolic predicted output token and \(y_i(A)\) the observed patched output token under candidate \(A\)'s sites.

For S2 and S3, the observed output alphabet is `\mathcal{N}_s \cup {OTHER}`. The symbolic prediction remains in `\mathcal{N}_s`. This is the locked closure rule for off-vocabulary patched outputs.

Primary residual code uses a symmetric noise model over the output vocabulary of size \(K_s\):
- fit the candidate’s validation error rate
\[
\hat e_A = \frac{1}{|D_{val}|}\sum_{i \in D_{val}} \mathbf{1}[\hat y_i(A) \neq y_i(A)],
\]
- set
\[
\epsilon_A = clip\!\left(\hat e_A,\ \epsilon_{min},\ 1 - \frac{1}{K_s} - \epsilon_{min}\right)
\]
with
\[
\epsilon_{min} = \max(10^{-4}, \frac{1}{10n_{val}})
\]

Then
\[
q_A(y_i \mid d_i(A)) =
\begin{cases}
1-\epsilon_A & \text{if } y_i(A) = \hat y_i(A) \\
\epsilon_A / (K_s-1) & \text{otherwise}
\end{cases}
\]

and
\[
L_{res}^{split}(A) = \sum_{d_i(A) \in D_{split}(A)} -\log_2 q_A(y_i \mid d_i(A))
\]

Primary split-specific values: `val`, `test`, `shift`.

### Implementation hooks
- fit \(\epsilon_A\) on validation only,
- compute test and shift residual bits without refitting,
- store both total bits and bits/example,
- store per-group residual-bit contributions so grouped bootstrap can resample candidate and null residual totals without rerunning the model.

## R10. Total bits and Pareto view

### Intuition
Selection uses total bits; visualization uses the residual-vs-structure tradeoff.

### Precise mechanism
Primary scalar objective on test:
\[
L_{tot}^{test}(A) = L_{struct}(A) + L_{res}^{test}(A)
\]

Primary visualization:
- x-axis: \(L_{struct}(A)\)
- y-axis: \(L_{res}^{test}(A)\)
- lower-left is better.

Bits/example:
\[
\bar L_{tot}^{test}(A) = L_{tot}^{test}(A) / n_{test}
\]

### Implementation hooks
- save both total bits and bits/example,
- compute Pareto-efficient candidates over `(L_struct, L_res_test)`.

## R11. Null/control pool

### Intuition
Acceptance must be calibrated against matched spurious explanations, not only against other candidate hypotheses.

### Precise mechanism
Primary null families:
1. **random_site**: same \(H\), same \(b\), same map family/hypergrid; generate 64 uniformly random **full disjoint** site-group proposals directly, score them with the matched validation rule, and retain the best validation-residual proposal as the null candidate record for that `(H,m,h,b)` tuple.
2. **shuffled_pair**: same \(H\), same candidate search and training procedure; source/base pairings are shuffled within intervention type during candidate construction, breaking causal correspondence. Retain the best matched-budget proposal as the null candidate record for that `(H,m,h,b)` tuple.
3. **untrained_model**: same architecture but randomly initialized low-level model; run the same matched search and retain the best proposal as the null candidate record for that `(H,m,h,b)` tuple. Required for planted and miniature-transformer settings, optional for GPT-2-small due compute.

Additional adversarial test (not part of the primary null envelope in the real-model setting):
4. **wrong_H_competitor**: the other hand-written high-level algorithms in the candidate library.

Available primary null families by setting:
- S1 planted: `random_site`, `shuffled_pair`, `untrained_model`
- S2 miniature learned IOI: `random_site`, `shuffled_pair`, `untrained_model`
- S3 GPT-2-small IOI: `random_site`, `shuffled_pair` by default; include `untrained_model` only if it is actually run and logged

Shuffled-pair stage semantics are locked as follows:
- Build one deterministic shuffled tuple dataset per `(setting, split, intervention_type, template_family)` for `split in {train, val}` only, using `shuffled_pair_seed = 0`.
- Shuffling replaces the source member of each tuple with a permuted source from the same stratum. Base input and intervention type stay fixed.
- If every canonical `train` and `val` stratum for that setting has size `1`, treat `shuffled_pair` as unavailable for that run rather than silently scoring an identity "shuffle" as a null family.
- The same shuffled train/val datasets are reused for every `(H,m,h,b)` cell in that setting.
- Use shuffled train/val tuples for:
  - readout fitting,
  - single-site prescreen,
  - variable-level group proposal construction,
  - full-candidate validation scoring,
  - one-pass local refinement.
- Final `test` scoring always uses the real unshuffled test tuples.
- Final `shift` scoring always uses the real unshuffled shift tuples.
- Grouped bootstrap and null-frontier rebuilding use the saved unshuffled `test` residual-bit contributions of the fitted shuffled-pair null candidates. Do not reshuffle inside bootstrap.

### Implementation hooks
- label null-family membership in saved artifacts,
- keep wrong-H candidates in the candidate ranking table,
- use only the available primary null families for the primary null envelope,
- one scored null candidate record is produced per `(family, H, m, h, b)` tuple in v1.

## R12. Complexity and tuning-budget matching

### Intuition
Nulls must be given the same fitting opportunity, or the envelope is meaningless.

### Precise mechanism
For a null candidate to be matched, it must preserve:
- same low-level setting,
- same map family,
- same hypergrid,
- same per-variable site budget \(b\),
- same number of training epochs / early-stop logic,
- same optimizer family,
- same number of random restarts,
- same train/val/test/shift splits,
- same candidate-generation budget.

Locked tuning budget details:
- linear families: 1 deterministic fit per proposal,
- `mlp1_relu`: 3 random restarts per proposal, choose best validation NLL,
- candidate generation uses the locked 64-proposal budget per `(H,m,h,b)`,
- `random_site` nulls sample 64 full disjoint random site-group proposals per `(H,m,h,b)`,
- `shuffled_pair` nulls use the same 64-proposal search and one-pass refinement budget,
- `untrained_model` nulls use the same search budget on the untrained model.

Null search budget must equal candidate search budget exactly.

### Implementation hooks
- use one shared search function with a `mode` flag,
- save the actual search budget in run metadata,
- do not under-tune nulls.

## R13. Null-frontier estimation

### Intuition
The null envelope should be transparent, stable, and not dominated by one prolific null family.

### Precise mechanism
For each setting and evaluation split:

1. determine the available primary null families for the setting,
2. pool null candidates from those families,
3. bin by structural bits using **2-bit-wide** bins,
4. inside each bin, compute family counts and define
   \[
   k_j = \min_{f \in \mathcal F_s} count(f, j),
   \]
   where \(\mathcal F_s\) is the available family set,
5. a bin is **valid** iff \(k_j \ge 5\),
6. in each valid bin, deterministically choose a balanced subset of size \(k_j\) per family using a fixed `frontier_balance_seed = 0`,
7. compute the empirical **5th percentile** of residual bits in the balanced subset using `np.quantile(..., 0.05, method="linear")`,
8. fit isotonic regression with `increasing=False` over valid bin centers and those 5th-percentile values,
9. store the valid-bin mask and bin centers, and define the frontier only for candidates whose structural bits fall inside a valid 2-bit bin; no extrapolation outside valid bins and no interpolation across invalid bins.

Call the result:
\[
\widehat L_{null,res}^{split}(x)
\]

The candidate gap is:
\[
G^{split}(A) = \widehat L_{null,res}^{split}(L_{struct}(A)) - L_{res}^{split}(A)
\]

Positive gap means the candidate compresses better than the best balanced nulls at comparable complexity.

Clarification:
- frontier eligibility is **bin-based**, not center-interval-based,
- map \(L_{struct}(A)\) to its 2-bit structural bin,
- if that bin is valid, evaluate the frontier at that bin's center and treat the frontier as defined for \(A\),
- if that bin is invalid, the frontier is undefined for that split.

If \(L_{struct}(A)\) falls in an invalid frontier bin on `test` or `shift`, then the frontier is undefined for that split and \(A\) is automatically **not support-eligible** on that split.

### Implementation hooks
- store raw null points and binned summary points,
- store the balanced candidate IDs used per bin,
- store the valid-bin mask and frontier domain summary metadata,
- use no extrapolation outside valid bins and no interpolation across invalid bins,
- if frontier is undefined for a candidate, record `support_reason = frontier_undefined` rather than guessing.

## R14. Acceptance rule

### Intuition
Acceptance should require both good total bits and a positive advantage over matched nulls.

### Precise mechanism
A candidate abstraction \(A\) is **supported** iff all are true:
1. \(A\) is in the candidate pool (not a null),
2. the null frontier is defined for \(A\) on both `test` and `shift`,
3. \(\bar L_{tot}^{test}(A)\) is within **0.01 bits/example** of the best candidate test total bits,
4. the one-sided 95% grouped-bootstrap lower confidence bound for \(G^{test}(A)\) is \(> 0\),
5. \(G^{shift}(A) > 0\) without refitting.

Grouped-bootstrap rule (locked):
- unit of resampling: grouped base/source tuples from the `test` split,
- number of replicates: `B_boot = 1000`,
- for each replicate:
  1. resample test group IDs with replacement,
  2. recompute candidate \(A\)'s resampled residual bits from saved per-group residual contributions using the fixed validation-fit \(\epsilon_A\),
  3. recompute every null candidate’s resampled residual bits on the same resampled group multiset,
  4. rebuild the test null frontier from those resampled null residual totals using the same valid bins, balanced subset IDs, quantile rule, and isotonic settings,
  5. compute \(G^{test,b}(A)\),
- the lower confidence bound is the empirical 5th percentile of \(\{G^{test,b}(A)\}_{b=1}^{B_{boot}}\).

The paper reports the set of supported candidates, not a forced single winner.

### Implementation hooks
- bootstrap over grouped intervention tuples, not individual rows,
- frontier validity is determined from the full null candidate pool and reused in bootstrap,
- keep test and shift acceptance separate in outputs,
- if no candidate is supported, positive framing fails for that setting.

## R15. Equivalence-class handling

### Intuition
The method should not overclaim uniqueness if multiple abstractions remain essentially tied.

### Precise mechanism
The **reported abstraction class** for a setting is:
- all supported candidates whose test total bits are within 0.01 bits/example of the best supported candidate.

If more than one candidate is supported, report the full class and the within-class ordering.

### Implementation hooks
- produce a compact table of all supported candidates,
- never collapse to one abstraction unless the class is singleton.

## R16. Robustness code variant

### Intuition
The primary codebook must survive at least one substantially different parameter-code approximation.

### Precise mechanism
Required robustness variant:
- keep \(L(H)\), \(L(b)\), \(L(S)\), \(L(m)\), \(L(h)\) unchanged,
- replace the BIC-style parameter code with an explicit quantized parameter code:

\[
L_{\theta}^{quant}(A) = 16 \cdot p_{eff}(A)
\]

after standardized inputs and fixed parameter clipping to \([-8, 8]\) before quantization.

No prequential code is required in v1.

### Implementation hooks
- implement codebook selection as a config switch,
- compare rankings and support decisions under primary and robustness codebooks.

## R17. Minimum viable empirical package

### Locked minimum package
1. **Planted finite-state setting** with known true abstraction.
2. **Miniature learned IOI transformer** trained from scratch on templated data with the same abstract schema.
3. **GPT-2-small IOI** with fixed-length prompt families and single-token names.

This is the minimum publishable package. Do not add more tasks until these are solid.

## R18. Positive-framing invalidation criteria

Positive framing is invalidated if any of these happen:
1. expressive maps do not create a meaningful fit-alone problem in the locked abstraction family,
2. the primary criterion fails to recover the planted true abstraction,
3. conclusions reverse under the required robustness code variant,
4. no candidate in GPT-2-small achieves a positive test null-gap lower CI and a positive shift gap.

If positive framing fails, pivot to a negative paper on the limits of control-calibrated acceptance in this abstraction family.

## Edge cases and failure modes
- fixed-length prompt validation can fail because of tokenizer behavior,
- null frontiers can be sparse in high-bit bins,
- sparse models can report unstable `nnz` counts without proper thresholding,
- a large supported equivalence class weakens uniqueness but does **not** invalidate the paper,
- no supported candidate in the real-model case is a serious risk and must be reported honestly.

## Wrong shortcuts Codex must not take
- do not change the output from token identity to correctness-only labels,
- do not collapse the candidate library to output heuristics with no internal variables,
- do not swap grouped-tuple bootstraps for row-wise bootstraps,
- do not use only random-site nulls,
- do not replace held-out residual bits with training loss,
- do not skip the shift requirement.
