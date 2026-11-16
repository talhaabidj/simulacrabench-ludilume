"""SimulacraBench submission v92a6: a per-item stack chosen on the visible rows.

For every scored item several conditional models of the answer given the
always-visible GIVEN block compete:

  - the crowd marginal (the official baseline),
  - multinomial logistic regression (main effects, and with the most
    informative column pairs added and shrunk harder),
  - Dirichlet back-off over the two, three or four GIVEN columns that carry
    the most information about the item, and over columns chosen greedily
    for what they add given the others (interactions, counted),
  - a vote of the nearest visible respondents over the whole GIVEN profile,
  - one multi-task network over every item at once (shared structure),
  - gradient-boosted trees with categorical splits (learned interactions),
  - TabICL v2, when a GPU and the bundled checkpoint are present.

Each one predicts every visible row out of fold. Mixture weights are then
fitted by EM to maximise the grader's own floored log score on those
out-of-fold predictions, the models are refitted on all visible rows, and the
held-out rows get the weighted mix. A model that does not help an item ends up
with a weight near zero, so nothing here is tuned on the cells being scored.

Skip logic is used where it is a fact rather than a guess: when an item's gate
parent is GIVEN and every visible respondent obeys the rule, closed rows get the
sentinel outright and open rows are modelled without it.

The run goes in passes so a short clock costs accuracy, not answers: every
item first gets the cheap candidates and the network, then trees (two folds),
regression and ladders, and the dearer candidates are added item by item,
worst-fitted first, while time remains. Any
failure falls back to the crowd marginal, so the run always returns legal
vectors.
"""

import contextlib
import ctypes
import hashlib
import os
import platform
import re
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

_IMPORTED = time.time()  # the platform's clock covers the import and all three instruments

try:
    from sklearn.linear_model import LogisticRegression
except Exception:                      # no sklearn: no regression candidate
    LogisticRegression = None
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
except Exception:
    HistGradientBoostingClassifier = None
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


def _preload_openmp():
    """Load the bundled OpenMP runtime when the system has none.

    LightGBM's Linux wheel needs the system libgomp, which slim images lack;
    without it the import fails and the trees quietly fall back to sklearn's.
    The copies next to this file are Debian's libgomp (GCC runtime library
    exception; see libgomp-copyright.txt), loaded globally so the wheel's
    library finds the symbols it asks for.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        ctypes.CDLL("libgomp.so.1")
        return                         # the system has one
    except OSError:
        pass
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "libgomp-%s.so.1" % platform.machine())
    try:
        if os.path.exists(path):
            ctypes.CDLL(path, mode=os.RTLD_GLOBAL)
    except OSError:
        pass


_preload_openmp()
try:
    import lightgbm                    # pinned in requirements.txt
except Exception:                      # not installed: sklearn's trees instead
    lightgbm = None

SEED = 0
FOLDS = 4                # out-of-fold predictions for the stack
FLOOR = 1e-3             # the grader's floor, used to score candidates
LR_C = 0.3               # inverse L2 strength of the regression candidate
PAIRS = 3                # column pairs added to the interaction regression
PAIR_SHRINK = 0.5        # pair columns scaled down: 4x the L2 penalty on their weights
MAX_ITER = 300
BACKOFF = ((2, 2.0), (3, 3.0), (4, 6.0))   # (GIVEN columns, pseudo-counts per level)
# Greedy back-off: (most columns, pseudo-counts, discount). A discount above
# zero is Kneser-Ney style: each count gives up a little, and the coarser level
# gets more weight the more distinct answers a group has seen.
LADDERS = ((4, 3.0, 0.0), (4, 1.0, 0.5))
POOL_MIN_ROWS = 200      # fewer fit rows: plain linear blend only
# Per item, a temperature on the linear blend and a log blend are tried out of
# fold. It was off after Dev submission 1026020 lost two thirds of UNHCR, but
# a Kaggle run at full size later showed that was a timeout, not the blend.
BLEND_CHOICE = True
UNFLOOR = True           # send the floor's best reply when it scores better out of fold (v55)
# Held-out blend weight of the three candidates that carry the blend, as a multiple of their fitted weight.
TILT = {"pfn35_full": 3.0, "network": 3.0, "icl": 1.5, "pfn35_more": 3.0, "icl_more": 1.5}
TILT_NEEDS_FULL_FIT = ("icl", "pfn35_full", "icl_more", "pfn35_more")   # raised only where their held-out prediction came from a fit on all rows, as measured
UNFLOOR_MARGIN = 0.0     # nats per row it must gain there
TEMPERATURES = np.exp(np.linspace(np.log(0.5), np.log(2.0), 15))
TREES = dict(max_iter=100, learning_rate=0.1, max_leaf_nodes=15,
             min_samples_leaf=20, l2_regularization=1.0, early_stopping=True,
             validation_fraction=0.1, n_iter_no_change=5, random_state=SEED)
# Deeper trees for routing rules that combine many GIVEN answers; early
# stopping decides how far they go.
TREES_DEEP = dict(TREES, max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                  min_samples_leaf=5, n_iter_no_change=10)
# A third setting: wider trees with little smoothing of rare categories, so
# the blend can choose between settings on the real rows (LightGBM only).
TREES_WIDE = dict(TREES, max_iter=200, learning_rate=0.05, max_leaf_nodes=63,
                  min_samples_leaf=10, n_iter_no_change=10,
                  lightgbm=dict(cat_smooth=1.0, cat_l2=1.0, min_data_per_group=5))
# Trees for a child of a scored parent also see the parent's predicted
# distribution, so partial routing (gates agree with the data at a median of
# 0.83 on UNHCR) is learned rather than assumed.
TREES_PARENT = dict(TREES, max_iter=150, learning_rate=0.08)
TREES_ENCODED = dict(TREES, max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                     lightgbm=dict(feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, seed=1))
TE_ALPHA = 10.0          # target encoding: shrinkage toward the overall shares, in respondents
PARENT_ALPHA = 2.0       # parent-mixture: pseudo-counts toward the child's marginal
USE_FOUNDATION = True    # ablation switch: TabICL and TabPFN off when False (a Dev reading of their worth)
# Lean stack: no TabPFN, no deep, wide, tuned or encoded trees, no regression and
# no ordinal candidate. On real data and the sandbox these took about 60% of the
# candidate time for about 7% of the blend weight (REVIEW.md section 52); the time
# goes to the network instead.
LEAN = False
# Pilot: after the plain trees, each optional family (deep and encoded trees,
# wide trees, regression, pair regression, TabPFN) is tried on a few items of
# this instrument's own rows, and runs on the rest only if it lowered the
# held-out loss of the mix there by more than PILOT_MIN_GAIN nats per row
# (_Stack.gain_of). The data in the container decides, not the sandbox.
# Off from v52: on bench run 6 v50, v51 and v51 without the pilot were within
# 0.00003 nats per cell of each other (REVIEW.md section 56).
PILOT = False
PILOT_ITEMS = 6
PILOT_MIN_GAIN = 2e-4
PILOT_FAMILY_SHARE = 0.03   # a family's trial stops here; one too slow for its trial is too slow to run
USE_GATE_CANDIDATES = True   # ablation switch: no parent, router, chain or branch candidates when False
ROUTERS = 6              # scored items, besides the gate parents, whose answers route others
ROUTER_WIDTH = 6         # a router has at most this many options, sentinel included
RULE_AGREEMENT = 0.98    # a GIVEN gate rule is used when this share of visible rows obeys it
RULE_MIN_ROWS = 50       # and at least this many visible rows show it (1 / (1 - RULE_AGREEMENT))
# LightGBM grows the same trees in compiled code that releases the GIL, so
# items fit in parallel threads (sklearn's trees run a Python loop per round
# and gain nothing from threads). On sandbox UNHCR it was 2x faster per fit,
# 2.45x faster again on four threads, and its held-out loss was lower.
LIGHTGBM = dict(min_data_per_group=20, cat_smooth=10.0, cat_l2=10.0, num_threads=1,
                force_col_wise=True, deterministic=True, verbose=-1)
SMALL_FIT = 200          # fewer fit rows: no early stopping, a few rounds only
NEIGHBOURS = 48          # nearest visible respondents kept per row, by GIVEN profile
KERNEL_PRIOR = 0.5       # how many respondents the crowd marginal is worth there
MIN_ROWS = 50            # fewer visible rows than this: marginal only
# Time. The platform allows 900 s in phase 1 and 3600 s in phase 2 for the
# import and all three instruments together. Each call takes its share of
# what is left, in proportion to how much work the instruments still to come
# need, so time one instrument leaves unused passes to the next.
TOTAL_DEV = 900.0
TOTAL_TEST = 3600.0
RESERVE = 80.0           # kept back for the driver's own work before and after
KNOWN_ITEMS = (12, 76, 213)   # scored items of the three public schemas
BUDGET = 150.0           # an instrument's nominal need: this plus
BUDGET_PER_ITEM = 1.1    # this per scored item,
BUDGET_MAX = 390.0       # at most this
# Slower practice machines stretch the clock (SB_BUDGET_SCALE); the platform
# sets neither variable, so there it is 1.0 and eight threads.
SCALE = float(os.environ.get("SB_BUDGET_SCALE", "1"))
_CALLED = []             # scored item counts of the instruments already called
# The run goes in passes so every item gets a model inside the budget: fast
# candidates for all items, then trees, then regression and ladders, then the
# dear candidates. Each later pass starts no new item after its share.
PARENTS_SHARE = 0.35     # gate parents get the tree and regression candidates first
TREES_SHARE = 0.53       # shallow trees on every item: the lever proven on real UNHCR data
TREES_FLOOR = 0.25       # and at least this share of the budget from the moment the pass starts,
                         # whatever the earlier passes took (213 UNHCR items need about 0.22 on a T4)
TREES2_SHARE = 0.74      # deep and target-encoded trees on every item, before anything unmeasured on Dev
ICL_SHARE = 0.79         # TabICL in what the trees leave, routers first (unmeasured on Dev)
# TabICL runs on the GPU in a background thread while the CPU fits the trees,
# instead of waiting for them. On real public surveys it took 13 to 53 percent
# of the blend weight where it ran, but the serial order left it 14 s (v46).
ICL_CONCURRENT = True
GATE_SHARE = 0.89        # router-informed candidates for children of scored parents (needs about a tenth)
DEEP_SHARE = 0.91        # wide trees (and the tuned setting in Test)
MEDIUM_SHARE = 0.93      # regression and ladders
DEAR_SHARE = 0.94        # pair regression
META_SHARE = 0.97        # the stacking meta-learner is tried while this share of the budget lasts
STRATA_MAX = 4           # blend weights per level of a GIVEN column with at most this many levels
STRATA_COLUMNS = 3       # at most this many such columns tried, fewest levels first
STRATA_SHRINK = 50.0     # rows' worth of pull from a level's weights toward the global weights
META_MIN_ROWS = 1000     # fewer fit rows: the EM blend only
META_C = 0.3             # inverse L2 strength of the meta-learner
JOBS = int(os.environ.get("SB_JOBS", min(8, os.cpu_count() or 1)))
NET_HIDDEN = 256         # multi-task network: width of the shared trunk
NET_DROPOUT = 0.2
NET_LR = 2e-3
NET_DECAY = 1e-4
NET_BATCH = 256
NET_EPOCHS = 60          # ceiling; folds stop early on their held-out loss
NET_PATIENCE = 4
EM_PRIOR = 0.5           # Dirichlet prior on the blend weights, in pseudo-rows per candidate
NET_ITEM_EPOCHS = True   # each item keeps the network's predictions from its own best epoch
NET_HONEST_OOF = True    # out-of-fold rows get the epoch chosen on the other part of the held half
NET_SHARE = 0.30         # the network must finish inside this share of the budget (0.25 up to v48; no longer tied to PILOT)
NET_SEEDS = (0, 1, 2, 3)  # networks averaged; no seed starts that would miss the deadline
NET_B = dict(hidden=512, dropout=0.35, lr=1e-3)   # a second architecture, Test only: the sweep's best settings
NET_SEEDS_B = (10, 11)
NET_C = dict(members=16)  # TabM-mini (Apache-2.0 idea, our own forty lines), Test only, one seed
NET_SEEDS_C = (20,)
NET_SHARE_TEST = 0.35    # in Test the second architecture trains inside this share
TREES_TUNED = dict(TREES, max_iter=400, learning_rate=0.05, max_leaf_nodes=7, min_samples_leaf=40)  # Test only: a more regularised setting for the blend to choose
ICL_REPO = "jingang/TabICL"                            # BSD-3, commit 4dcd344
ICL_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"
ICL_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
ICL_ESTIMATORS = 1       # ensemble members in Dev: coverage beats members (proxy: one member costs 0.003 to 0.009 nats, two cover half the items)
ICL_ESTIMATORS_TEST = 2  # in Test: eight members cost 70 to 104 s per item on a T4 and reached fewer items than Dev did (rehearsal of v29); coverage first
ICL_MORE = 8             # members of a second TabICL candidate, added in Test in spare time only
PFN35_MORE = 4           # the same for TabPFN-3.5
MORE_SHARE = 0.50        # the extra-member candidates stop at this share of the budget; later passes keep their room
FAST_SHARE = 0.20        # and start only if both first passes finished inside this share (a T4 needs 0.26 to 0.79)
# TabICL needs a GPU on the platform; SB_ICL_CPU=1 lets practice runs use the CPU.
ICL_DEVICE = "cpu" if os.environ.get("SB_ICL_CPU") else "cuda"
PFN_REPO = "Prior-Labs/TabPFN-v2-clf"                  # Prior Labs License 1.1 (Apache 2.0 plus attribution)
PFN_CHECKPOINT = "tabpfn-v2-classifier-v2_default.ckpt"
PFN_SHA256 = "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
PFN_ESTIMATORS = 1       # TabPFN v2 members in Dev
PFN_ESTIMATORS_TEST = 2  # in Test (four cost 100 to 140 s per item on a T4)
PFN_MAX_CLASSES = 10     # TabPFN v2's hard limit; wider items do without it
PFN_SHARE = 0.79         # Dev: TabPFN only in what TabICL leaves of its window (it is slower and no better)
PFN_SHARE_TEST = 0.78    # Test: its own window after TabICL's
# NVIDIA Kumo-Tabular Large (OpenMDW 1.1). On the platform models.txt names the
# public repo, so the original weights arrive in the Hugging Face cache; they
# are used only if they hash to the published file. Practice machines without
# that cache may put our float16 re-save next to this file instead (never
# shipped in the zip). Runs after TabICL on each item in TabICL's window. The
# code is structured-data-models 0.2.1 (Apache-2.0), bundled as sdm/ so the
# build never has to upgrade torch for it.
USE_KUMO = True
REQUIRE_MODELS = os.environ.get("SB_REQUIRE_MODELS", "1") == "1"   # a real instrument without TabICL or TabPFN-3.5 raises
KUMO_ONLY = False        # bench switch: Kumo-Tabular replaces TabICL in its pass
KUMO_SIZE = "medium"     # large 215M parameters, medium 62M, small 28M; medium reached 3 to 15 items per dataset on a T4 (REVIEW.md section 53)
KUMO_REPO = "nvidia/Kumo-Tabular"
KUMO_FILES = {           # the published classifier checkpoints and their sha256
    "large": "cc21484316949641cc3d65a7bb6c36cdea854cc569435755ff23b850018ec613",
    "medium": "70d6aee8fd13416300afead71a5145a40edb790952e2417d87f27b3f7e57793d",
    "small": "1ff91f484e19021aaf7d9eff6cc07e6d95a60b4aaad473e83e2ab2c3bfd2b617",
}
KUMO_FP16_FILE = "kumo-tabular-v1.0.1-large-classifier-fp16.pt"
KUMO_FP16_SHA256 = "d79c907f0fb51f382e88ec7429c677b5eac43a212acac750c9f1205bfcaa4561"
KUMO_ESTIMATORS = 1      # ensemble members in Dev
KUMO_ESTIMATORS_TEST = 2


def _options(schema, item):
    record = schema["items"][item]
    return list(record["values"]) + (
        [schema["gated_value"]] if record.get("gate") else [])


def _codes(series, options):
    lookup = {value: i for i, value in enumerate(options)}
    return series.map(lookup).fillna(-1).astype(np.int64).to_numpy()


_NUMERIC = re.compile(r"^(?:(?:more than|less than|before)\s+|<=|>=|[<>])?\d+(\.\d+)?(\s*(?:-|to)\s*\d+(\.\d+)?)?\s*(\+|and above|or more)?$")
_NUMBERED = re.compile(r"^\s*(\d{1,2})\s*[.)]\s*\S")
NUMBERED_OTHER = 90      # "98. Don't know", "99. Refused" and the like sit outside the scale
# Items whose numbered labels have no order (read from the public schemas):
# they get no ordinal candidate and no smoothing between neighbouring codes.
NOMINAL_ITEMS = frozenset({
    "s2_6_return_decision_maker", "s2_10_house_condition_lbl", "s2_11_house_occupancy_lbl",
    "s3_3_who_visited_lbl", "s3_10d_employ_before_return_lbl", "s3_10g_employ_syria_lbl",
    "s3_15_friend_return_likely_lbl", "s3_23a_safety_rank1_lbl", "s3_23b_safety_rank2_lbl",
    "s4_5_main_earner_relation_lbl", "s4_16_employment_type_lbl",
    "s7a_5_alone_reason_lbl", "s7a_6_return_companions_lbl",
})
# Scales the label patterns miss or get wrong, in their real order (read from
# the public schemas). Months are listed as text there ("1", "10", "11", ...),
# date fields carry the non-date codes 98/99, and durations and likelihoods
# are words. "Under 6 months" overlaps the finer bands, so it stays off the scale.
_DURATIONS = ("Less than one month", "About one month", "2-3 months", "4-6 months",
              "6 months – 1 year", "More than 1 year", "More than 2 years", "More than 5 years")
ORDERED_SCALES = {
    **{item: tuple(str(month) for month in range(1, 13))
       for item in ("s3_4b_visit_month", "s3_8b_return_month")},
    **{item: ("before 2011", "2011-2014", "2015-2018", "2019-2022", "2023", "2024", "2025")
       for item in ("s3_4a_visit_year", "s3_8a_return_year")},
    **{item: _DURATIONS for item in ("user.howLongSearchingForJob", "user.howLongUnemployed",
                                     "user.howLongInCurrentPosition")},
    "tr_recommend_full": ("Not likely", "Not sure", "Likely"),
    "Q14": ("Not at all important", "Somehow Important", "Very Important"),
}
SMOOTHING = (0.1, 0.2, 0.35)   # share of each scale level's mass moved to its neighbours


def _numeric(value):
    """Is this level a number, a band ("18-24", "25 - 34") or an open band ("45+")?

    Numbered labels such as "1. Head of household" are not: relationship and
    marital status are coded with numbers but have no order.
    """
    text = str(value).strip().lower().replace("years old", "").replace("years", "")
    return bool(_NUMERIC.match(text.strip()))


def _numbered(value):
    """The code of a numbered label ("3. Once a week"), or None."""
    match = _NUMBERED.match(str(value))
    if not match or int(match.group(1)) >= NUMBERED_OTHER:
        return None
    return int(match.group(1))


def _scale(values, item=None):
    """Indices of an item's levels that form an ordered scale, or None.

    Banded scores ("<2.5", "2.5-3.0", ... "4.5-5.0") and counts are scales;
    a level such as "Not answered" or the gate sentinel stays outside it.
    Schema-reviewed nominal items have no scale. Items in ORDERED_SCALES use
    that order, without their off-scale codes. Other numbered
    labels ("1. Very safe" ... "5. Very unsafe") follow code order.
    """
    if item in NOMINAL_ITEMS:
        return None
    if item in ORDERED_SCALES:
        lookup = {value: i for i, value in enumerate(values)}
        scale = [lookup[value] for value in ORDERED_SCALES[item] if value in lookup]
        return np.asarray(scale, dtype=np.int64) if len(scale) >= 3 else None
    scale = [i for i, value in enumerate(values) if _numeric(value)]
    if len(scale) >= 3 and len(scale) >= len(values) - 2:
        return np.asarray(scale)
    numbered = [(code, i) for i, value in enumerate(values)
                if (code := _numbered(value)) is not None]
    if len(numbered) >= 3 and len({code for code, _ in numbered}) == len(numbered):
        return np.asarray([i for _, i in sorted(numbered)])
    return None


def _smooth(p, scale, lam):
    """Move a share lam of each scale level's mass to its two neighbours.

    Neighbouring levels of a banded score are near-substitutes, so a little
    mass spread to them costs less than it saves when the answer is one band
    off. Mass is conserved: an end level keeps what it cannot pass on.
    """
    if lam <= 0 or scale is None or len(scale) < 3:
        return p
    q = p.copy()
    s = p[:, scale]
    moved = np.zeros_like(s)
    moved[:, 1:] += 0.5 * lam * s[:, :-1]
    moved[:, :-1] += 0.5 * lam * s[:, 1:]
    kept = s * (1.0 - lam)
    kept[:, 0] += 0.5 * lam * s[:, 0]          # end levels keep the undeliverable half
    kept[:, -1] += 0.5 * lam * s[:, -1]
    q[:, scale] = kept + moved
    return q


def _order(values, code):
    """Position of each answer on an ordered scale, from 0 to 1, or None.

    Bands and counts ("18-24", "6-10", "11+") are ordered, so the regression
    and the network can share strength between neighbouring levels; one-hot
    columns alone treat them as unrelated. One level outside the scale, such
    as "[generalised:s2]" or "Not answered", sits at the middle.
    """
    scale = [i for i, value in enumerate(values) if _numeric(value)]
    if len(scale) < 3 or len(scale) < len(values) - 1:
        return None
    position = np.full(len(values) + 1, 0.5)
    position[scale] = np.arange(len(scale)) / (len(scale) - 1)
    return position[code]


def _smoothed(counts):
    weights = counts + 0.5
    return weights / weights.sum()


def _floored(p):
    k = p.shape[-1]
    p = p / p.sum(axis=-1, keepdims=True)
    return FLOOR + (1.0 - k * FLOOR) * p


def _unfloored(p, total):
    """The vector to send so that the grader's floored vector is p itself.

    The grader scores FLOOR + (1 - total * FLOOR) * sent over `total` options,
    so sending (p - FLOOR) / scale returns p exactly. Options p puts below the
    floor are sent as zero and the others give up their shortfall in
    proportion (water-filling), which is the best reply when p is the belief.
    Options past p's own width (an open gate's sentinel) are sent as zero.
    Returns (sent, scored): sent has p's width, scored is what the grader scores.
    """
    p = p / p.sum(axis=1, keepdims=True)
    width = p.shape[1]
    room = 1.0 - (total - width) * FLOOR       # what p's options share after the extra floors
    ordered = -np.sort(-p, axis=1)
    active = np.arange(1, width + 1)
    mu = np.cumsum(ordered, axis=1) / (room - (width - active) * FLOOR)
    fits = ordered >= FLOOR * mu               # the j largest options stay above the floor
    last = width - 1 - np.argmax(fits[:, ::-1], axis=1)
    mu = mu[np.arange(len(p)), last][:, None]
    scored = np.maximum(FLOOR, p / mu)
    sent = np.maximum(scored - FLOOR, 0.0) / (1.0 - total * FLOOR)
    return sent / sent.sum(axis=1, keepdims=True), scored


def _onehot(code, width):
    out = np.zeros((len(code), width))
    out[np.arange(len(code)), code] = 1.0
    return out


class _Context:
    """What every item shares: the GIVEN block in two encodings, and the folds."""

    def __init__(self, frame, schema, given, rng):
        n = len(frame)
        self.cards, codes, blocks = [], [], []
        for item in given:
            values = list(schema["items"][item]["values"])
            code = _codes(frame[item], values)
            code = np.where(code >= 0, code, len(values))   # anything else
            self.cards.append(len(values) + 1)
            codes.append(code)
            block = _onehot(code, len(values) + 1)
            blocks.append(block[:, block.sum(axis=0) > 0])
            rank = _order(values, code)
            if rank is not None:
                blocks.append(rank[:, None])
        self.z = np.stack(codes, axis=1) if codes else np.zeros((n, 0), np.int64)
        self.x = np.hstack(blocks) if blocks else np.ones((n, 1))
        self.household = None
        key_items = HOUSEHOLD_KEY + HOUSEHOLD_COUNTS
        if all(item in given for item in key_items):
            columns = self.z[:, [given.index(item) for item in key_items]]
            self.household = np.unique(columns, axis=0, return_inverse=True)[1].ravel()
        # Folds by a stable hash of the respondent id, so neither the folds nor
        # anything built on them depends on the order of the rows; the random
        # draw stays as the fallback when there is no id column.
        if "respondent_id" in frame.columns:
            digests = [hashlib.sha256(str(v).encode("utf-8")).digest()[:4] for v in frame["respondent_id"]]
            self.fold = np.array([int.from_bytes(d, "big") % FOLDS for d in digests], np.int64)
        else:
            self.fold = rng.integers(0, FOLDS, size=n)
        self.near = self.far = None
        self.votes = {}          # (fit, out) -> neighbours and weights, see vote_weights

    def vote_weights(self, fit, out, rate):
        """Each out row's neighbours among the fit rows and their kernel weights.

        Every item asks for the same few (fit, out) pairs, so the results are
        kept. Returns (neighbour rows, usable mask, weights); a neighbour
        outside the fit rows is not usable and weighs 0.
        """
        key = (hash(fit.tobytes()), hash(out.tobytes()), len(fit), len(out))
        if key not in self.votes:
            member = np.zeros(len(self.fold), bool)
            member[fit] = True
            near = self.near[out]
            usable = (near >= 0) & member[np.maximum(near, 0)]
            self.votes[key] = (np.maximum(near, 0), usable, {})
        near, usable, weights = self.votes[key]
        if rate not in weights:
            weights[rate] = np.where(usable, np.exp(-rate * np.where(usable, self.far[out], 0.0)), 0.0)
        return near, usable, weights[rate]

    def neighbours(self, visible, deadline, chunk=512):
        """The NEIGHBOURS visible rows closest to each row, by how many GIVEN
        answers differ. A visible row only looks outside its own fold, so the
        out-of-fold scores stay honest; a held-out row looks at every visible
        row. Fills self.near (row indices, -1 where missing) and self.far.
        """
        z = self.z.astype(np.int16)
        n, columns = z.shape
        if columns == 0 or visible.sum() == 0:
            return
        near = np.full((n, NEIGHBOURS), -1, np.int64)
        far = np.full((n, NEIGHBOURS), np.inf)
        groups = [(visible & (self.fold == k), visible & (self.fold != k))
                  for k in range(FOLDS)] + [(~visible, visible)]
        for queries, pool in groups:
            pool = np.flatnonzero(pool)
            queries = np.flatnonzero(queries)
            if len(pool) == 0:
                continue
            keep = min(NEIGHBOURS, len(pool))
            for start in range(0, len(queries), chunk):
                if time.time() > deadline:
                    raise TimeoutError
                rows = queries[start:start + chunk]
                distance = np.zeros((len(rows), len(pool)), np.int16)
                for g in range(columns):
                    distance += z[rows, g][:, None] != z[pool, g][None, :]
                best = np.argpartition(distance, keep - 1, axis=1)[:, :keep]
                near[rows, :keep] = pool[best]
                far[rows, :keep] = np.take_along_axis(distance, best, axis=1)
        self.near, self.far = near, far


# ---- candidates: each maps (fit rows, out rows) to (len(out), width) ------

def _marginal(ctx, fit, out, y, width):
    return np.tile(_smoothed(np.bincount(y[fit], minlength=width).astype(float)),
                   (len(out), 1))


def _regression(c):
    def run(ctx, fit, out, y, width):
        present = np.unique(y[fit])
        if len(present) == 1:
            return _onehot(np.full(len(out), present[0]), width)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(C=c, max_iter=MAX_ITER)
            model.fit(ctx.x[fit], y[fit])
            p = np.zeros((len(out), width))
            p[:, model.classes_] = model.predict_proba(ctx.x[out])
        return p
    run.label = "regression"
    return run


def _information(x, y, nx, ny):
    """Mutual information minus its small-sample bias, in nats."""
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
    n = joint.sum()
    p = joint / n
    expected = p.sum(axis=1, keepdims=True) * p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / expected[nz])).sum()
                 - (nx - 1) * (ny - 1) / (2.0 * n))


def _renumber(key_fit, key_out):
    """Strata as 0..n-1 on the fit rows; out rows in an unseen stratum get n."""
    seen, key_fit = np.unique(key_fit, return_inverse=True)
    where = np.searchsorted(seen, key_out)
    found = (where < len(seen)) & (seen[np.minimum(where, len(seen) - 1)] == key_out)
    return key_fit, np.where(found, where, len(seen)), len(seen)


def _backoff(columns, alpha, greedy=False, discount=0.0):
    """Counts in ever finer GIVEN strata, each level smoothed onto the coarser.

    The strata are the `columns` GIVEN items most informative about the item
    on the fit rows; `alpha` is how many respondents each level is worth. With
    `greedy`, each next column is the one that adds most information given
    the strata already chosen, so columns that only matter in combination
    (round x country routing) are found; otherwise columns are ranked one by
    one.
    """
    def run(ctx, fit, out, y, width):
        y_fit = y[fit]
        running = _marginal(ctx, fit, out, y, width)
        if ctx.z.shape[1] == 0:
            return running
        if greedy:
            # The first step also tries every pair, since two columns can
            # matter together while neither does alone.
            ladder, key, base = [], np.zeros(len(fit), np.int64), 0.0
            pair, pair_gain = None, 0.0
            pairs = _pair_gains(ctx, fit, y_fit, width) if ctx.z.shape[1] > 1 else []
            if pairs and pairs[0][0] > 0:
                pair_gain, g, h = pairs[0]
                pair = (g, h)
                pair_key, _, _ = _renumber(ctx.z[fit, g] * ctx.cards[h] + ctx.z[fit, h], key[:0])
            for _ in range(columns):
                best, best_gain, best_key = None, 0.0, None
                for g in range(ctx.z.shape[1]):
                    if g in ladder:
                        continue
                    trial, _, n = _renumber(key * ctx.cards[g] + ctx.z[fit, g], key[:0])
                    gain = _information(trial, y_fit, n, width) - base
                    if gain > best_gain:
                        best, best_gain, best_key = g, gain, trial
                if not ladder and pair is not None and pair_gain > best_gain:
                    ladder, key, base = list(pair), pair_key, pair_gain
                    continue
                if best is None or len(ladder) >= columns:
                    break
                ladder.append(best)
                key, base = best_key, base + best_gain
        else:
            score = [_information(ctx.z[fit, g], y_fit, ctx.cards[g], width)
                     for g in range(ctx.z.shape[1])]
            ladder = [g for g in np.argsort(score)[::-1][:columns] if score[g] > 0]
        key_fit = np.zeros(len(fit), np.int64)
        key_out = np.zeros(len(out), np.int64)
        for g in ladder:
            # Renumber the strata seen so far, so the table never grows with
            # the product of the cardinalities.
            key_fit, key_out, n = _renumber(key_fit * ctx.cards[g] + ctx.z[fit, g],
                                            key_out * ctx.cards[g] + ctx.z[out, g])
            table = np.bincount(key_fit * width + y_fit,
                                minlength=(n + 1) * width).reshape(-1, width)
            counts = table[key_out].astype(float)
            distinct = (counts > 0).sum(axis=1, keepdims=True)
            running = (np.maximum(counts - discount, 0.0)
                       + (alpha + discount * distinct) * running) / (
                counts.sum(axis=1, keepdims=True) + alpha)
        return running
    run.label = ("ladder%d" if greedy else "backoff%d") % columns + ("kn" if discount else "")
    return run


def _kernel(rate):
    """A vote of the nearest visible respondents by whole GIVEN profile.

    Each neighbour among the fit rows counts exp(-rate * columns that differ),
    and the crowd marginal adds KERNEL_PRIOR respondents' worth. Unlike the
    back-off, every GIVEN column takes part at once.
    """
    def run(ctx, fit, out, y, width):
        base = _marginal(ctx, fit, out[:1], y, width)[0]
        near, usable, weight = ctx.vote_weights(fit, out, rate)
        cells = np.arange(len(out))[:, None] * width + np.where(usable, y[near], 0)
        votes = np.bincount(cells.ravel(), weights=weight.ravel(),
                            minlength=len(out) * width).reshape(len(out), width)
        return (votes + KERNEL_PRIOR * base) / (
            weight.sum(axis=1, keepdims=True) + KERNEL_PRIOR)
    run.label = "kernel%g" % rate
    return run


TWIN_PRIOR = 0.1
PANEL = True             # cross-round household linkage (v92a)
PANEL_WAVE = "wave"
PANEL_BLOCK = "s1_2_country_asylum_lbl"   # partners share it; candidates are compared inside it
PANEL_PLACES = ("governorate_col", "s2_4a_origin_governorate_lbl")
PANEL_PEOPLE = ("s2_1_hh_relationship_lbl", "s2_1b_age", "s2_1c_education_level_lbl",
                "s2_1d_marital_status_lbl", "s2_2_single_parent_lbl")
PANEL_CLAIM_Z = 3.0      # visible rows link when each is the other's best by this many SDs
PANEL_CLAIM_GAP = 1.0    # and the next best trails by at least this many SDs
PANEL_MIN_CLAIMS = 30    # fewer links: no panel candidates
PANEL_TOP = 8            # candidate partners kept per row
PANEL_ALPHA = 2.0        # pseudo-counts toward the round's marginal in a carry-over table
PANEL_RHO_MAX = 0.95     # at most this chance that a row's partner is in the frame
PANEL_EM_ROUNDS = 100    # EM rounds for that chance
PANEL_U_PAIRS = 200000   # random cross-round pairs that give the non-match rates
PANEL_ROUNDS = 1         # v92a6: claim passes; each links the rows the last one left free
PANEL_GIVEN_TAU = 4.0    # v92a6: GIVEN log likelihood ratio a looser link needs (None: no such stage)
PANEL_CLAIM_Z2 = 1.5     # v92a6: answer z of that looser stage
PANEL_CLAIM_GAP2 = 1.0   # v92a6: and its runner-up gap
HOUSEHOLD_KEY = ("s1_2_country_asylum_lbl", "governorate_col", "s2_4a_origin_governorate_lbl")
HOUSEHOLD_COUNTS = ("s2_3_boys_0_5", "s2_3_girls_0_5", "s2_3_boys_6_17", "s2_3_girls_6_17",
                    "s2_3_men_18_59", "s2_3_women_18_59", "s2_3_men_60plus", "s2_3_women_60plus")

def _twins_household(ctx, fit, out, y, width):
    """Fit-only counts among equal GIVEN profiles; rate-3 fallback."""
    fallback = _kernel(3.0)(ctx, fit, out, y, width)
    base = _marginal(ctx, fit, out[:1], y, width)[0]
    groups = int(ctx.household.max()) + 1
    votes = np.bincount(ctx.household[fit] * width + y[fit], minlength=groups * width).reshape(groups, width).astype(float)
    own = votes[ctx.household[out]]
    count = own.sum(axis=1, keepdims=True)
    return np.where(count > 0, (own + TWIN_PRIOR * base) / (count + TWIN_PRIOR), fallback)

_twins_household.label = "twins_hh"


def _panel_note(text):
    """A line for SB_DEBUG's .panel file; nothing is written on the platform."""
    if os.environ.get("SB_DEBUG"):
        with open(os.environ["SB_DEBUG"] + ".panel", "a", encoding="utf-8") as fh:
            fh.write(text + "\n")


def _panel_unknown(label):
    text = str(label).lower()
    return ("generalis" in text or "want to answer" in text or "know" in text
            or text in ("nan", "none", "__missing__", "__other__"))


def _panel_count(label):
    text = str(label).strip()
    if text.isdigit():
        return float(text)
    found = re.match(r"^(\d+)\s*-\s*(\d+)$", text)
    return (float(found.group(1)) + float(found.group(2))) / 2 if found else np.nan


def _panel_fields(schema, given, ctx):
    """Per row: the codes of the place and person columns with an 'unknown' flag,
    and the roster counts as numbers (NaN where a band has no number)."""
    def coded(item):
        values = list(schema["items"][item]["values"])
        code = ctx.z[:, given.index(item)]
        unknown = np.array([_panel_unknown(v) for v in values] + [True])
        return code, unknown[code]
    counts = []
    for item in HOUSEHOLD_COUNTS:
        numbers = np.array([_panel_count(v) for v in schema["items"][item]["values"]] + [np.nan])
        counts.append(numbers[ctx.z[:, given.index(item)]])
    return {"places": [coded(item) for item in PANEL_PLACES],
            "people": [coded(item) for item in PANEL_PEOPLE],
            "counts": np.stack(counts, axis=1)}


def _panel_levels(fields, q, c):
    """Agreement levels of row pairs (q, c), index arrays that broadcast: a place
    level (each place same / different / unknown, 9 in all) and a household level
    (roster: exact, same size, size off by 1, by 2, other; times how many of the
    answering adult's columns agree, 0 to 5)."""
    shape = np.broadcast(q, c).shape
    place = np.zeros(shape, np.int64)
    for code, unknown in fields["places"]:
        level = np.where(unknown[q] | unknown[c], 2, np.where(code[q] == code[c], 0, 1))
        place = place * 3 + level
    diff = fields["counts"][q] - fields["counts"][c]
    missing = np.isnan(diff).any(axis=-1)
    gap = np.abs(np.where(np.isnan(diff), 0.0, diff).sum(axis=-1))
    exact = (diff == 0).all(axis=-1)
    roster = np.select([missing, exact, gap == 0, gap <= 1, gap <= 2], [4, 0, 1, 2, 3], 4)
    agree = np.zeros(shape, np.int64)
    for code, unknown in fields["people"]:
        agree += (code[q] == code[c]) & ~unknown[q] & ~unknown[c]
    return place, roster * (len(fields["people"]) + 1) + agree


def _panel_claim(onehot, weight, answered, visible, side, block, claimed_by, z_min, gap_min, allowed=None):
    """Link free visible rows across rounds inside each block: each is the other's best (among
    allowed pairs) by at least z_min SDs of its row and its column, and the runner-up trails by at
    least gap_min SDs. Marks claimed_by and returns the new pairs."""
    pairs = []
    for value in np.unique(block):
        a = np.flatnonzero(visible & (side == 0) & (block == value) & (claimed_by < 0))
        b = np.flatnonzero(visible & (side == 1) & (block == value) & (claimed_by < 0))
        if len(a) < 3 or len(b) < 3:
            continue
        overlap = np.maximum(answered[a] @ answered[b].T, 1.0)
        score = ((onehot[a] * weight) @ onehot[b].T) / overlap
        row_mean, row_sd = score.mean(axis=1), score.std(axis=1) + 1e-6
        col_mean, col_sd = score.mean(axis=0), score.std(axis=0) + 1e-6
        ranked = score if allowed is None else np.where(allowed(a, b), score, -np.inf)
        best_b, best_a = ranked.argmax(axis=1), ranked.argmax(axis=0)
        second_row = np.partition(ranked, -2, axis=1)[:, -2]
        second_col = np.partition(ranked, -2, axis=0)[-2]
        for i, j in enumerate(best_b):
            if best_a[j] != i or not np.isfinite(ranked[i, j]):
                continue
            s = score[i, j]
            zr, zc = (s - row_mean[i]) / row_sd[i], (s - col_mean[j]) / col_sd[j]
            if min(zr, zc) < z_min:
                continue
            if min((s - second_row[i]) / row_sd[i], (s - second_col[j]) / col_sd[j]) < gap_min:
                continue
            claimed_by[a[i]], claimed_by[b[j]] = b[j], a[i]
            pairs.append((a[i], b[j], row_sd[i], col_sd[j], zr, zc, s, overlap[i, j], z_min))
    return pairs


def _panel_build(frame, schema, given, ctx, jobs, visible):
    """Link households across the two rounds; None when the columns are missing
    or too few visible pairs link. See experiments/build_v92a.py for the design."""
    needed = (PANEL_WAVE, PANEL_BLOCK) + PANEL_PLACES + HOUSEHOLD_COUNTS + PANEL_PEOPLE
    if any(item not in given for item in needed):
        return None
    n = len(frame)
    wave = frame[PANEL_WAVE].astype(str).to_numpy()
    known = frame[PANEL_WAVE].notna().to_numpy()
    rounds = sorted(set(wave[known]))
    if len(rounds) != 2:
        return None
    side = np.full(n, -1, np.int64)
    side[known & (wave == rounds[0])] = 0
    side[known & (wave == rounds[1])] = 1
    block = ctx.z[:, given.index(PANEL_BLOCK)]

    # Visible answers, one column per option; a pair's score counts the items it agrees on.
    widths = [job[2] for job in jobs]
    offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int64)
    onehot = np.zeros((n, int(offsets[-1])), np.float32)
    for j, job in enumerate(jobs):
        ok = np.flatnonzero(visible & (job[1] >= 0))
        onehot[ok, offsets[j] + job[1][ok]] = 1.0
    share = onehot[visible].mean(axis=0)
    weight = np.where(share > 0, 1.0, 0.0).astype(np.float32)   # plain agreement count (link diagnostic run1)
    for j, job in enumerate(jobs):    # v92a5: only answered cells count (not gate-closed, missing or other)
        for k, value in enumerate(_options(schema, job[0])):
            if value in ("__MISSING__", "__OTHER__") or (job[3] and k == job[2] - 1):
                weight[offsets[j] + k] = 0.0
    answered = np.stack([onehot[:, offsets[j]:offsets[j + 1]] @ weight[offsets[j]:offsets[j + 1]]
                         for j in range(len(jobs))], axis=1)

    claimed_by = np.full(n, -1, np.int64)
    pairs = []
    for _ in range(PANEL_ROUNDS):
        found = _panel_claim(onehot, weight, answered, visible, side, block, claimed_by,
                             PANEL_CLAIM_Z, PANEL_CLAIM_GAP)
        pairs += found
        if not found:
            break
    if len(pairs) < PANEL_MIN_CLAIMS:
        _panel_note("panel off: %d linked pairs" % len(pairs))
        return None
    pairs = np.array(pairs)
    pa, pb = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)

    # GIVEN agreement on linked pairs (m) against random cross-round pairs (u).
    fields = _panel_fields(schema, given, ctx)
    places, households = 9, 5 * (len(PANEL_PEOPLE) + 1)
    mp, mh = _panel_levels(fields, pa, pb)
    rng = np.random.default_rng(SEED)
    total = sum(int(((side == 0) & (block == v)).sum()) * int(((side == 1) & (block == v)).sum())
                for v in np.unique(block))
    uq, uc = [], []
    for value in np.unique(block):
        a = np.flatnonzero((side == 0) & (block == value))
        b = np.flatnonzero((side == 1) & (block == value))
        if len(a) and len(b):
            k = int(PANEL_U_PAIRS * len(a) * len(b) / max(total, 1)) + 1
            uq.append(rng.choice(a, k))
            uc.append(rng.choice(b, k))
    up, uh = _panel_levels(fields, np.concatenate(uq), np.concatenate(uc))

    def log_ratio(m, u, width):
        m = np.bincount(m, minlength=width) + 0.5
        u = np.bincount(u, minlength=width) + 0.5
        return np.log(m / m.sum()) - np.log(u / u.sum())
    lp, lh = log_ratio(mp, up, places), log_ratio(mh, uh, households)

    if PANEL_GIVEN_TAU is not None:   # v92a6: looser answer links where GIVEN agrees too
        def allowed(a, b):
            ok = np.zeros((len(a), len(b)), bool)
            for start in range(0, len(a), 256):
                place, household = _panel_levels(fields, a[start:start + 256, None], b[None, :])
                ok[start:start + 256] = lp[place] + lh[household] >= PANEL_GIVEN_TAU
            return ok
        more = _panel_claim(onehot, weight, answered, visible, side, block, claimed_by,
                            PANEL_CLAIM_Z2, PANEL_CLAIM_GAP2, allowed)
        if more:
            pairs = np.vstack([pairs, np.array(more)])
            pa, pb = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)

    # Each row's candidates: the other round in its block, less rows linked to someone else.
    # A candidate's prior is rho / (number of candidates); mean_log is the log of the
    # candidates' average likelihood ratio, from which rho is fitted below.
    top = np.full((n, PANEL_TOP), -1, np.int64)
    top_log = np.full((n, PANEL_TOP), -np.inf)
    mean_log = np.full(n, -np.inf)
    for value in np.unique(block):
        for s in (0, 1):
            q = np.flatnonzero((side == s) & (block == value))
            c = np.flatnonzero((side == 1 - s) & (block == value))
            if not len(q) or not len(c):
                continue
            keep = min(PANEL_TOP, len(c))
            taken = claimed_by[c]
            for start in range(0, len(q), 256):
                rows = q[start:start + 256]
                place, household = _panel_levels(fields, rows[:, None], c[None, :])
                logl = lp[place] + lh[household]
                blocked = (taken[None, :] >= 0) & (taken[None, :] != rows[:, None])
                logl = np.where(blocked, -np.inf, logl)
                size = np.log(np.maximum(np.isfinite(logl).sum(axis=1), 1))
                peak = logl.max(axis=1)
                found = np.isfinite(peak)
                safe = np.where(found, peak, 0.0)
                total = safe + np.log(np.exp(logl - safe[:, None]).sum(axis=1)) - size
                mean_log[rows] = np.where(found, total, -np.inf)
                best = np.argpartition(-logl, keep - 1, axis=1)[:, :keep]
                top_log[rows, :keep] = np.take_along_axis(logl, best, axis=1) - size[:, None]
                top[rows, :keep] = np.where(np.isfinite(top_log[rows, :keep]), c[best], -1)

    # rho per round: the share of rows whose partner is among their candidates, fitted by EM on
    # every row of the round (answer links find too few pairs to count it from them).
    rho = np.zeros(2)
    post = np.zeros((n, PANEL_TOP))
    for s in (0, 1):
        rows = np.flatnonzero((side == s) & np.isfinite(mean_log))
        if not len(rows):
            continue
        ratio = np.exp(np.minimum(mean_log[rows], 50.0))
        share = 0.5
        for _ in range(PANEL_EM_ROUNDS):
            share = float(np.mean(share * ratio / (share * ratio + 1.0 - share)))
            share = min(max(share, 1e-4), PANEL_RHO_MAX)
        rho[s] = share
        scale = share / (share * ratio + 1.0 - share)
        post[rows] = np.where(np.isfinite(top_log[rows]),
                              np.exp(np.minimum(top_log[rows], 50.0)) * scale[:, None], 0.0)
    _panel_note("panel on: %d linked pairs (%d answer-only), rho %.3f/%.3f, held-out mass %.3f, visible mass %.3f"
                % (len(pairs), len(mp), rho[0], rho[1], post[~visible].sum(axis=1).mean() if (~visible).any() else 0.0,
                   post[visible].sum(axis=1).mean()))
    if os.environ.get("SB_DEBUG") and "respondent_id" in frame.columns:
        ids = frame["respondent_id"].astype(str).to_numpy()
        np.savez(os.environ["SB_DEBUG"] + ".panel.npz", ids=ids, pa=pa, pb=pb, top=top, post=post,
                 side=side, visible=visible, rho=rho, lp=lp, lh=lh, z=pairs[:, 4:6])
    return {"side": side, "pa": pa, "pb": pb, "sd_a": pairs[:, 2], "sd_b": pairs[:, 3],
            "zr": pairs[:, 4], "zc": pairs[:, 5], "weight": weight, "offsets": offsets,
            "score": pairs[:, 6], "overlap": pairs[:, 7], "need": pairs[:, 8],
            "index": {job[0]: j for j, job in enumerate(jobs)}, "top": top, "post": post}


def _panel_candidates(panel, item, base_oof, base_final):
    """The partner's answer as two candidates over the network's prediction.

    A carry-over table per direction, P(this round's answer | partner's answer),
    is counted on linked pairs whose rows are both fit rows; a pair is left out
    for this item if it links only thanks to this item's own agreement.
    panel_mix mixes the partners' carried-over answers into the network's
    prediction by their probabilities; panel_lr multiplies the network's
    prediction by the likelihood ratio the partners' answers give.
    """
    j = panel["index"][item]
    side, pa, pb = panel["side"], panel["pa"], panel["pb"]

    def tables(y, fit, width):
        member = np.zeros(len(side), bool)
        member[fit] = True
        cell_a = panel["weight"][panel["offsets"][j] + y[pa]]
        cell_b = panel["weight"][panel["offsets"][j] + y[pb]]
        agree = cell_a * (y[pa] == y[pb])
        both = cell_a * cell_b
        without = (panel["score"] * panel["overlap"] - agree) / np.maximum(panel["overlap"] - both, 1.0)
        own = panel["score"] - without    # how much this item's cell adds to the pair's score
        robust = np.minimum(panel["zr"] - own / panel["sd_a"], panel["zc"] - own / panel["sd_b"]) >= panel["need"]
        keep = member[pa] & member[pb] & robust
        out = {}
        for s in (0, 1):
            here = fit[side[fit] == s]
            q = _smoothed(np.bincount(y[here], minlength=width).astype(float))
            source, target = (pa, pb) if s == 1 else (pb, pa)
            counts = np.zeros((width, width))
            np.add.at(counts, (y[source[keep]], y[target[keep]]), 1.0)
            out[s] = ((counts + PANEL_ALPHA * q) / (counts.sum(axis=1, keepdims=True) + PANEL_ALPHA), q)
        return out, member

    def make(kind):
        def run(ctx, fit, out, y, width, final_fit=False):
            base = (base_final if final_fit else base_oof)[out][:, :width]
            base = base / base.sum(axis=1, keepdims=True)
            table, member = tables(y, fit, width)
            partners = panel["top"][out]
            usable = (partners >= 0) & member[np.maximum(partners, 0)]
            weights = np.where(usable, panel["post"][out], 0.0)
            answers = np.clip(y[np.maximum(partners, 0)], 0, width - 1)
            p = base.copy()
            for s in (0, 1):
                rows = side[out] == s
                if not rows.any():
                    continue
                carry_table, q = table[s]
                carry = np.einsum("rk,rkc->rc", weights[rows], carry_table[answers[rows]])
                rest = 1.0 - weights[rows].sum(axis=1, keepdims=True)
                if kind == "mix":
                    p[rows] = rest * base[rows] + carry
                else:
                    p[rows] = base[rows] * (rest + carry / q[None, :])
            return p / p.sum(axis=1, keepdims=True)
        run.label = "panel_" + kind
        run.fixed = True
        return run
    return [make("mix"), make("lr")]


def _key_information(key, y, keys, width):
    """Mutual information of y with a key in 0..keys-1, minus its small-sample
    bias. Counts with one bincount; keys never seen are left out of the bias."""
    joint = np.bincount(key * width + y, minlength=keys * width).reshape(keys, width)
    joint = joint[joint.sum(axis=1) > 0].astype(float)
    n = joint.sum()
    p = joint / n
    expected = p.sum(axis=1, keepdims=True) * p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / expected[nz])).sum()
                 - (len(joint) - 1) * (width - 1) / (2.0 * n))


def _pair_gains(ctx, fit, y_fit, width):
    """(information, g, h) for every pair of GIVEN columns, best first."""
    scored = []
    for g in range(ctx.z.shape[1]):
        for h in range(g + 1, ctx.z.shape[1]):
            key = ctx.z[fit, g] * ctx.cards[h] + ctx.z[fit, h]
            scored.append((_key_information(key, y_fit, ctx.cards[g] * ctx.cards[h], width), g, h))
    scored.sort(reverse=True)
    return scored


def _best_pairs(ctx, fit, y, width, count):
    """The `count` pairs of GIVEN columns that carry most information about y."""
    return [(g, h) for gain, g, h in _pair_gains(ctx, fit, y[fit], width)[:count] if gain > 0]


def _pair_columns(ctx, pairs, fit, rows):
    """One-hot columns for each pair's combined value, numbered on the fit rows.

    A combination never seen in the fit rows gets an all-zero row, so it falls
    back to the main effects.
    """
    from scipy import sparse
    blocks = []
    for g, h in pairs:
        key_fit = ctx.z[fit, g] * ctx.cards[h] + ctx.z[fit, h]
        key_rows = ctx.z[rows, g] * ctx.cards[h] + ctx.z[rows, h]
        _, code, n = _renumber(key_fit, key_rows)
        seen = code < n
        blocks.append(sparse.csr_matrix(
            (np.ones(seen.sum()), (np.flatnonzero(seen), code[seen])), shape=(len(rows), n)))
    return sparse.hstack(blocks, format="csr")


def _regression_pairs(c, count, shrink):
    """Logistic regression on the main effects plus the most informative pairs.

    Pair columns are scaled by `shrink`, which under the L2 penalty shrinks
    their weights harder than the main effects: interactions are pooled more,
    as in multilevel regression.
    """
    def run(ctx, fit, out, y, width):
        from scipy import sparse
        present = np.unique(y[fit])
        if len(present) == 1:
            return _onehot(np.full(len(out), present[0]), width)
        pairs = _best_pairs(ctx, fit, y, width, count)
        x_fit, x_out = sparse.csr_matrix(ctx.x[fit]), sparse.csr_matrix(ctx.x[out])
        if pairs:
            x_fit = sparse.hstack([x_fit, shrink * _pair_columns(ctx, pairs, fit, fit)], format="csr")
            x_out = sparse.hstack([x_out, shrink * _pair_columns(ctx, pairs, fit, out)], format="csr")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(C=c, max_iter=MAX_ITER)
            model.fit(x_fit, y[fit])
            p = np.zeros((len(out), width))
            p[:, model.classes_] = model.predict_proba(x_out)
        return p
    run.label = "regression_pairs"
    return run


def _lightgbm(params, z_fit, y_fit, z_out, classes, categorical=None):
    """LightGBM with categorical splits; y_fit holds classes 0..classes-1.

    `categorical` lists the categorical columns; by default every column is.
    """
    settings = dict(LIGHTGBM, objective="multiclass", num_class=classes, seed=SEED,
                    learning_rate=params["learning_rate"],
                    num_leaves=params["max_leaf_nodes"],
                    min_data_in_leaf=params["min_samples_leaf"],
                    lambda_l2=params["l2_regularization"])
    settings.update(params.get("lightgbm", {}))
    columns = list(range(z_fit.shape[1])) if categorical is None else list(categorical)
    if len(y_fit) < SMALL_FIT:
        rounds, callbacks, valid = max(params["max_iter"] // 4, 1), [], []
        train = lightgbm.Dataset(z_fit, y_fit, categorical_feature=columns)
    else:
        held = np.random.default_rng(SEED).random(len(y_fit)) < params["validation_fraction"]
        train = lightgbm.Dataset(z_fit[~held], y_fit[~held], categorical_feature=columns)
        valid = [lightgbm.Dataset(z_fit[held], y_fit[held], categorical_feature=columns,
                                  reference=train)]
        rounds = params["max_iter"]
        callbacks = [lightgbm.early_stopping(params["n_iter_no_change"], verbose=False)]
    booster = lightgbm.train(settings, train, num_boost_round=rounds,
                             valid_sets=valid, callbacks=callbacks)
    return booster.predict(z_out, num_iteration=booster.best_iteration or None)


def _histgb(params, z_fit, y_fit, z_out, classes, categorical=None):
    """sklearn's trees, when LightGBM is missing; y_fit holds 0..classes-1."""
    is_categorical = np.ones(z_fit.shape[1], bool)
    if categorical is not None:
        is_categorical[:] = False
        is_categorical[list(categorical)] = True
    # Early stopping holds out a stratified tenth, which fails outright when
    # an option was chosen only once. Those few rows stay out of the tree
    # fit, and a region too small to hold out one row per option is fitted
    # without early stopping.
    keep = np.bincount(y_fit, minlength=classes)[y_fit] >= 2
    z_fit, y_fit = z_fit[keep], y_fit[keep]
    present = np.unique(y_fit)
    p = np.zeros((len(z_out), classes))
    if len(present) == 1:
        p[:, present[0]] = 1.0
        return p
    held = int(np.ceil(params["validation_fraction"] * len(y_fit)))
    stop = held >= len(present) and len(y_fit) - held >= len(present)
    model = HistGradientBoostingClassifier(
        categorical_features=is_categorical,
        **{k: v for k, v in dict(params, early_stopping=stop).items() if k != "lightgbm"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(z_fit, y_fit)
        p[:, model.classes_] = model.predict_proba(z_out)
    return p


def _target_encode(z_fit, y_fit, z_out, cards, classes, fold):
    """Each GIVEN column as its smoothed per-level answer shares, one column
    per answer option.

    Fit rows are encoded out of fold (counts from the other folds), so the
    trees that read these columns never see a row's own answer; out rows
    use every fit row. Shrinkage toward the overall shares is TE_ALPHA
    respondents' worth per level. This is the ordered-target-statistics idea
    of CatBoost, done once per candidate instead of per tree.
    """
    n_fit, columns = z_fit.shape
    prior = np.bincount(y_fit, minlength=classes).astype(float) + 0.5
    prior /= prior.sum()
    enc_fit = np.zeros((n_fit, columns * classes))
    enc_out = np.zeros((len(z_out), columns * classes))

    def table(z, y, card):
        counts = np.bincount(z * classes + y, minlength=card * classes).reshape(card, classes).astype(float)
        totals = counts.sum(axis=1, keepdims=True)
        return (counts + TE_ALPHA * prior[None, :]) / (totals + TE_ALPHA)

    for g in range(columns):
        card = cards[g]
        full = table(z_fit[:, g], y_fit, card)
        enc_out[:, g * classes:(g + 1) * classes] = full[z_out[:, g]]
        for k in np.unique(fold):
            held = fold == k
            if held.all():
                enc_fit[held, g * classes:(g + 1) * classes] = prior[None, :]
                continue
            part = table(z_fit[~held, g], y_fit[~held], card)
            enc_fit[held, g * classes:(g + 1) * classes] = part[z_fit[held, g]]
    return enc_fit, enc_out


def _trees_encoded(params, label):
    """LightGBM on the GIVEN codes plus their target encodings, with row and
    column subsampling and another seed, so it errs differently from the
    plain trees. Two folds, like the deep trees."""
    def run(ctx, fit, out, y, width):
        present, codes = np.unique(y[fit], return_inverse=True)
        if len(present) == 1:
            return _onehot(np.full(len(out), present[0]), width)
        grow = _lightgbm if lightgbm is not None else _histgb
        enc_fit, enc_out = _target_encode(ctx.z[fit], codes, ctx.z[out], ctx.cards,
                                          len(present), ctx.fold[fit])
        x_fit = np.hstack([ctx.z[fit].astype(float), enc_fit])
        x_out = np.hstack([ctx.z[out].astype(float), enc_out])
        p = np.zeros((len(out), width))
        p[:, present] = grow(params, x_fit, codes, x_out, len(present),
                             categorical=range(ctx.z.shape[1]))
        return p
    run.label = label
    run.halves = True
    return run


def _ordinal(params, scale, label="ordinal"):
    """An ordered-levels candidate for scale items (Frank and Hall, 2001).

    The multiclass trees give the off-scale levels ("98. Don't know", the
    sentinel) their mass; the scale levels share the rest through K-1 binary
    boosters of P(level > k | GIVEN), each fitted on the rows that answered
    on the scale. Differences of those cumulative probabilities, clipped and
    renormalised, are the level probabilities, so the order of the levels is
    part of the fit rather than only a smoothing afterwards.
    """
    scale = np.asarray(scale)

    def run(ctx, fit, out, y, width):
        present, codes = np.unique(y[fit], return_inverse=True)
        p = np.zeros((len(out), width))
        grow = _lightgbm if lightgbm is not None else _histgb
        if len(present) == 1:
            return _onehot(np.full(len(out), present[0]), width)
        full = np.zeros((len(out), width))
        full[:, present] = grow(params, ctx.z[fit], codes, ctx.z[out], len(present))
        on_scale = np.isin(y[fit], scale)
        rows = fit[on_scale]
        rank = {level: i for i, level in enumerate(scale)}
        r = np.array([rank[v] for v in y[rows]])
        levels = np.unique(r)
        if len(rows) < MIN_ROWS or len(levels) < 2:
            return full
        off = [i for i in range(width) if i not in rank]
        above = np.ones((len(out), len(scale)))        # P(rank > k), with P(rank > last) = 0
        above[:, -1] = 0.0
        for k in range(len(scale) - 1):
            target = (r > k).astype(np.int64)
            if target.min() == target.max():
                above[:, k] = float(target[0])
                continue
            q = grow(params, ctx.z[rows], target, ctx.z[out], 2)
            above[:, k] = q[:, 1]
        above = np.minimum.accumulate(above, axis=1)  # cumulative probabilities must not rise
        probs = np.diff(np.hstack([np.ones((len(out), 1)), above]), axis=1) * -1.0
        probs = np.clip(probs, 1e-6, None)
        probs /= probs.sum(axis=1, keepdims=True)
        share = 1.0 - full[:, off].sum(axis=1) if off else np.ones(len(out))
        p[:, off] = full[:, off]
        p[:, scale] = probs * share[:, None]
        return p
    run.label = label
    run.halves = True
    return run


def _trees(params, label, halves=False):
    """Gradient-boosted trees with categorical splits on the GIVEN codes:
    LightGBM when installed, else sklearn's.

    With `halves`, the stack fits them on two folds only and averages the two
    fold models for the final prediction: two fits per item instead of five.
    """
    def run(ctx, fit, out, y, width):
        present, codes = np.unique(y[fit], return_inverse=True)
        if len(present) == 1:
            return _onehot(np.full(len(out), present[0]), width)
        grow = _lightgbm if lightgbm is not None else _histgb
        p = np.zeros((len(out), width))
        p[:, present] = grow(params, ctx.z[fit], codes, ctx.z[out], len(present))
        return p
    run.label = label
    run.halves = halves
    return run


_ICL = {}                # the TabICL network, loaded once per process
_FULL_FINAL = {}         # {"on": True} in Test: in-context learners get a full-context final fit


def _icl_checkpoint():
    """Path to the pinned TabICL weights, or None.

    Bundled next to this file, or prefetched by the platform from `models.txt`
    into the Hugging Face cache; never downloaded here. Either way the file
    must hash to the pinned commit's checkpoint.
    """
    candidates = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ICL_CHECKPOINT)]
    try:
        from huggingface_hub import hf_hub_download
        candidates.append(hf_hub_download(ICL_REPO, ICL_CHECKPOINT, local_files_only=True))
    except Exception:
        pass
    for path in candidates:
        if path and os.path.exists(path):
            digest = hashlib.sha256()
            with open(path, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 22), b""):
                    digest.update(block)
            if digest.hexdigest() == ICL_SHA256:
                return path
    return None


def _icl_ready():
    """TabICL runs only with a GPU (or SB_ICL_CPU) and the pinned checkpoint on disk."""
    if "ok" in _ICL:
        return _ICL["ok"]
    _ICL["ok"] = False
    if not USE_FOUNDATION:
        return False
    try:
        import threading
        import torch
        from tabicl import TabICLClassifier
        path = _icl_checkpoint() if (ICL_DEVICE == "cpu" or torch.cuda.is_available()) else None
        if path is not None:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)

            from tabicl._model.tabicl import TabICL
            network = TabICL(**checkpoint["config"])
            network.load_state_dict(checkpoint["state_dict"])
            network.eval()                 # in training mode the output width is wrong

            class Preloaded(TabICLClassifier):
                """Uses the one network built above.

                Building a network per item cost a few tenths of a second
                each. One network is safe because every fit and predict
                happens under _ICL["lock"], so no two items are ever in
                flight on it at once; the network itself keeps no state
                between a fit and its predict except the KV cache, which is
                off.
                """
                def _load_model(self):
                    self.model_path_ = path
                    self.model_ = network
                    self.model_config_ = checkpoint["config"]

            _ICL.update(cls=Preloaded, lock=threading.Lock(), ok=True, network=network)
    except Exception:
        _ICL["ok"] = False
    return _ICL["ok"]


def _icl(ctx, fit, out, y, width, members=None):
    """In-context prediction by TabICL v2 on the GIVEN codes, one call at a time."""
    present = np.unique(y[fit])
    if len(present) == 1:
        return _onehot(np.full(len(out), present[0]), width)
    with _ICL["lock"]:
        # Workers queue on the lock; one that gets it after the pass's share
        # has run out gives up, so the queue cannot overrun the clock.
        if time.time() > _ICL.get("more_until" if members else "until", np.inf):
            raise TimeoutError
        model = _ICL["cls"](n_estimators=members or _ICL.get("estimators", ICL_ESTIMATORS),
                            device=ICL_DEVICE, allow_auto_download=False, random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(ctx.z[fit].astype(np.float32), y[fit])
            proba = model.predict_proba(ctx.z[out].astype(np.float32))
    p = np.zeros((len(out), width))
    p[:, np.asarray(model.classes_, dtype=np.int64)] = proba
    return p


_icl.label = "icl"
_icl.halves = True       # two folds: two in-context calls per item in Dev
_icl.full_final = True   # a third, full-context call for the final prediction in Test


def _icl_more(ctx, fit, out, y, width):
    """TabICL again with more ensemble members, as a candidate of its own."""
    return _icl(ctx, fit, out, y, width, members=ICL_MORE)


_icl_more.label = "icl_more"
_icl_more.halves = True
_icl_more.full_final = True


_PFN = {}                # TabPFN v2: class, checkpoint path and lock, set once per process


def _pfn_ready():
    """TabPFN v2 runs with a GPU (or SB_ICL_CPU) and its pinned checkpoint next to this file."""
    if "ok" in _PFN:
        return _PFN["ok"]
    _PFN["ok"] = False
    if not USE_FOUNDATION:
        return False
    try:
        import threading
        import torch
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
        from tabpfn import TabPFNClassifier
        if ICL_DEVICE != "cpu" and not torch.cuda.is_available():
            return False
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PFN_CHECKPOINT)
        if not os.path.exists(path):
            return False
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 22), b""):
                digest.update(block)
        if digest.hexdigest() != PFN_SHA256:
            return False
        _PFN.update(cls=TabPFNClassifier, path=path, lock=threading.Lock(), ok=True)
    except Exception:
        _PFN["ok"] = False
    return _PFN["ok"]


PFN_MIN_ROWS = 2         # TabPFN v2 breaks on a class with a single fit row; such classes are left out


def _common(y, fit, least):
    """The fit rows whose class has at least `least` fit rows."""
    counts = np.bincount(y[fit])
    kept = fit[counts[y[fit]] >= least]
    return kept if len(kept) else fit


def _pfn(ctx, fit, out, y, width):
    """In-context prediction by TabPFN v2 on the GIVEN codes, one call at a time.

    A second foundation model with a different pre-training prior than
    TabICL's; the blend decides what it is worth. Items with more than ten
    answer options are its hard limit and do without it.
    """
    fit = _common(y, fit, PFN_MIN_ROWS)
    present = np.unique(y[fit])
    if len(present) == 1:
        return _onehot(np.full(len(out), present[0]), width)
    if width > PFN_MAX_CLASSES:
        raise ValueError("more classes than TabPFN v2 supports")
    with _PFN["lock"]:
        if time.time() > _PFN.get("until", np.inf):
            raise TimeoutError
        model = _PFN["cls"](model_path=_PFN["path"], device=ICL_DEVICE,
                            n_estimators=_PFN.get("estimators", PFN_ESTIMATORS),
                            random_state=SEED, ignore_pretraining_limits=True,
                            categorical_features_indices=list(range(ctx.z.shape[1])))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(ctx.z[fit].astype(np.float32), y[fit])
            proba = model.predict_proba(ctx.z[out].astype(np.float32))
    p = np.zeros((len(out), width))
    p[:, np.asarray(model.classes_, dtype=np.int64)] = proba
    return p


_pfn.label = "pfn"
_pfn.halves = True
_pfn.full_final = True
# CUDA inference changes slightly with query-batch shape. Keep the historical
# query rows for this candidate; other candidates omit unused training queries.
_pfn.preserve_query_batch = True


_KUMO = {}               # Kumo-Tabular: the loaded model and its lock, set once per process
KUMO_REVISION = "c5893d418e162b9f841ae179dc767afefff73831"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _kumo_checkpoint(here):
    """The original weights from the models.txt cache, else a local copy, else None.

    A local copy is the original file named kumo-tabular-<size>-classifier.pt,
    or for the large model our float16 re-save.
    """
    expected = KUMO_FILES[KUMO_SIZE]
    try:
        from huggingface_hub import hf_hub_download
        # A revision-pinned prefetch creates a snapshot, but need not create
        # refs/main. The exact snapshot works in that cache layout. The main
        # fallback also handles organizer prefetch at a newer revision whose
        # model bytes are unchanged; every path still needs the pinned hash.
        for revision in (KUMO_REVISION, None):
            try:
                path = hf_hub_download(KUMO_REPO, KUMO_SIZE + "/classifier.pt",
                                       revision=revision, local_files_only=True)
                if _sha256(path) == expected:
                    return path
            except Exception:
                continue
    except Exception:
        pass
    local = [(os.path.join(here, "kumo-tabular-%s-classifier.pt" % KUMO_SIZE), expected)]
    if KUMO_SIZE == "large":
        local.append((os.path.join(here, KUMO_FP16_FILE), KUMO_FP16_SHA256))
    for path, digest in local:
        if os.path.exists(path) and _sha256(path) == digest:
            return path
    return None



PFN35_FILE = "tabpfn-v3.5-20260909_multiclass.safetensors"
PFN35_DIGEST = "4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149"
PFN35_REVISION = "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e"
# Bundled in the zip (scripts/package.py packs *_CHECKPOINT files and checks *_SHA256).
PFN35_CHECKPOINT = "tabpfn-v3.5-20260909_multiclass.safetensors"
PFN35_SHA256 = "4f1e155b09ae1c9860c75cd0a16f811b7431ddac4cb8022ee5e4498070f44149"


def _suffixed(path):
    """The weights under a name ending in .safetensors.

    The Hugging Face cache links the name to a blob without a suffix, and
    tabpfn picks the file format from the resolved suffix. A hard link (or a
    copy) under the real name keeps it.
    """
    if os.path.realpath(path).endswith(".safetensors"):
        return path
    import shutil
    import tempfile
    named = os.path.join(tempfile.mkdtemp(), PFN35_FILE)
    try:
        os.link(os.path.realpath(path), named)
    except OSError:
        shutil.copyfile(os.path.realpath(path), named)
    return named


def _kumo_ready():
    """Legacy scheduling slot; now loads only the full PFN3.5 candidate."""
    if "ok" in _KUMO:
        return _KUMO["ok"]
    _KUMO["ok"] = False
    if not (USE_FOUNDATION and USE_KUMO):
        return False
    try:
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        from huggingface_hub import hf_hub_download
        if ICL_DEVICE != "cpu" and not torch.cuda.is_available():
            return False
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TABPFN_MODEL_CACHE_SIZE", "2")
        here = os.path.dirname(os.path.abspath(__file__))
        paths = [os.path.join(here, PFN35_FILE)]
        for revision in (PFN35_REVISION, "main"):
            try:
                paths.append(hf_hub_download("Prior-Labs/tabpfn_3_5", PFN35_FILE,
                             revision=revision, local_files_only=True))
            except Exception:
                pass
        path = next((p for p in paths if os.path.isfile(p) and _sha256(p) == PFN35_DIGEST), None)
        if path is None:
            return False
        path = _suffixed(path)
        _KUMO.update(cls=TabPFNClassifier, version=ModelVersion.V3_5, path=path,
                     lock=threading.Lock(), ok=True)
    except Exception:
        _KUMO["ok"] = False
    return _KUMO["ok"]


def _kumo(ctx, fit, out, y, width, members=None):
    """Full multiclass prior. Preserve absent-class columns and sparse labels."""
    present = np.unique(y[fit])
    if len(present) == 1:
        return _onehot(np.full(len(out), present[0]), width)
    until = _KUMO.get("more_until" if members else "until", np.inf)
    with _KUMO["lock"]:
        if time.time() > until:
            raise TimeoutError
        model = _KUMO["cls"].create_default_for_version(
            _KUMO["version"], model_path=_KUMO["path"], device=ICL_DEVICE,
            n_estimators=members or 1, random_state=0,
            categorical_features_indices=list(range(ctx.z.shape[1])))
        model.fit(ctx.z[fit].astype(np.float32), y[fit])
        if time.time() > until:
            raise TimeoutError
        probabilities = model.predict_proba(ctx.z[out].astype(np.float32))
        classes = np.asarray(model.classes_, dtype=np.int64)
    if not np.array_equal(np.sort(classes), present):
        raise ValueError("Unexpected PFN3.5 classes")
    p = np.zeros((len(out), width))
    p[:, classes] = probabilities
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("Illegal PFN3.5 probability vectors")
    return p / p.sum(1, keepdims=True)


_kumo.label = "pfn35_full"
_kumo.halves = True
_kumo.full_final = True
_kumo.always_full_final = True


def _kumo_more(ctx, fit, out, y, width):
    """TabPFN-3.5 again with more ensemble members, as a candidate of its own."""
    return _kumo(ctx, fit, out, y, width, members=PFN35_MORE)


_kumo_more.label = "pfn35_more"
_kumo_more.halves = True
_kumo_more.full_final = True
_kumo_more.always_full_final = True


class _Probe:
    """A tiny made-up table with three classes, for checking that a model runs."""
    def __init__(self):
        self.z = np.random.default_rng(SEED).integers(0, 3, size=(60, 4))
        self.y = self.z[:, 0] % 3


def _models_work():
    """Fit and predict once with TabICL and TabPFN-3.5; raise if either fails."""
    probe = _Probe()
    fit, out = np.arange(40), np.arange(40, 60)
    checks = [("TabICL", _icl, _ICL)] + ([("TabPFN-3.5", _kumo, _KUMO)] if USE_KUMO else [])
    for name, model, state in checks:
        state["until"] = np.inf          # a deadline left by an earlier instrument must not apply
        try:
            p = model(probe, fit, out, probe.y, 3)
        except Exception as exc:
            raise RuntimeError(f"{name} could not fit and predict: {exc!r}") from exc
        if p.shape != (len(out), 3) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-4):
            raise RuntimeError(f"{name} returned invalid probabilities")


def _require_success(stacks, strict):
    """Raise if TabICL or TabPFN-3.5 failed on every item it tried.

    With `strict` (Test) also if it ends in the blend of no item for any
    other reason, time-outs included: a failed run can be sent again, a
    finished one without these models would be final. Dev stays lenient:
    on a slow practice GPU its short clock leaves TabPFN-3.5 a handful of
    UNHCR items, and none is not a fault there.
    """
    expected = ([] if KUMO_ONLY else [("icl", "TabICL")]) + ([("pfn35_full", "TabPFN-3.5")] if USE_KUMO else [])
    for label, name in expected:
        used = broken = 0
        for _, regions in stacks.values():
            for stack in regions:
                used += label in stack.labels
                broken += ("failed:" + label) in stack.failed
        if broken and not used:
            raise RuntimeError(f"{name} failed on every item it tried")
        if strict and not used:
            raise RuntimeError(f"{name} reached no item")


def _parent_mix(parent_p, parent_code):
    """A child item predicted through its scored parent.

    parent_p holds the parent's predicted distribution for every row (out of
    fold on visible rows, the current blend on held-out rows) and parent_code
    the parent's actual answer on visible rows. P(child | parent answer) is
    counted on the fit rows and shrunk toward the child's marginal; the child's
    prediction is that table mixed over the parent's distribution. It pools the
    routing of a whole family of items instead of learning it per item.
    """
    def run(ctx, fit, out, y, width):
        kp = parent_p.shape[1]
        base = _marginal(ctx, fit, out[:1], y, width)[0]
        table = np.bincount(parent_code[fit] * width + y[fit],
                            minlength=kp * width).reshape(kp, width).astype(float)
        table = (table + PARENT_ALPHA * base) / (table.sum(axis=1, keepdims=True) + PARENT_ALPHA)
        return parent_p[out] @ table
    run.label = "parent_mix"
    return run


def _chain(router_p, router_code, params, features, label="chain", given=None, cache=None):
    """The gate chain, learned: the sentinel through a router, the rest from trees.

    The child's sentinel probability is the router's predicted distribution
    passed through P(sentinel | router answer), counted on the fit rows, so
    every child of one router inherits that router's whole stack. The router
    is the declared parent, or the scored item whose answer says most about
    whether this one was asked. With `given` (a GIVEN column index) the rate
    table is P(sentinel | router answer, that column's value), shrunk toward
    the router-only rate: the survey round or the country can change who
    was asked. The other options come from trees fitted on the fit rows that
    were asked, with the routers' distributions (features) among their
    inputs; `cache` lets several chains of one item share those trees. This
    is the organisers' baseline's form with our models inside it.
    """
    def run(ctx, fit, out, y, width):
        sentinel = width - 1
        y_fit = y[fit]
        kp = router_p.shape[1]
        is_sentinel = (y_fit == sentinel).astype(float)
        counts = np.bincount(router_code[fit], weights=is_sentinel, minlength=kp)
        totals = np.bincount(router_code[fit], minlength=kp).astype(float)
        rate = (counts + PARENT_ALPHA * is_sentinel.mean()) / (totals + PARENT_ALPHA)
        if given is None:
            s = router_p[out] @ rate
        else:
            card = ctx.cards[given]
            cell = router_code[fit] * card + ctx.z[fit, given]
            counts2 = np.bincount(cell, weights=is_sentinel, minlength=kp * card).reshape(kp, card)
            totals2 = np.bincount(cell, minlength=kp * card).reshape(kp, card).astype(float)
            rate2 = (counts2 + PARENT_ALPHA * rate[:, None]) / (totals2 + PARENT_ALPHA)
            s = np.einsum("rv,vr->r", router_p[out], rate2[:, ctx.z[out, given]])
        s = np.clip(s, 0.0, 1.0)
        key = (hash(fit.tobytes()), hash(out.tobytes())) if cache is not None else None
        if key is not None and key in cache:
            p = cache[key].copy()
        else:
            asked = fit[y_fit != sentinel]
            p = np.zeros((len(out), width))
            if len(asked) < MIN_ROWS or len(np.unique(y[asked])) < 2:
                counts = np.bincount(y[asked], minlength=width)[:sentinel].astype(float)
                p[:, :sentinel] = _smoothed(counts)[None, :]
            else:
                present, codes = np.unique(y[asked], return_inverse=True)
                grow = _lightgbm if lightgbm is not None else _histgb
                x_fit = np.hstack([ctx.z[asked].astype(float), features[asked]])
                x_out = np.hstack([ctx.z[out].astype(float), features[out]])
                p[:, present] = grow(params, x_fit, codes, x_out, len(present),
                                     categorical=range(ctx.z.shape[1]))
            if key is not None:
                cache[key] = p.copy()
        p[:, :sentinel] *= (1.0 - s)[:, None]
        p[:, sentinel] = s
        return p
    run.label = label
    run.halves = True
    return run


def _branch_codes(jobs, parent_of, visible, schema):
    """For each gate parent with several children, which branch each visible
    respondent took, as one code.

    Children are grouped by their declared `observed_if` (the UNHCR return
    branches s7a, s7b and s7c are three such groups); a group counts as asked
    when most of its children were. The schema says the branches are
    mutually exclusive and the file only partly follows the declared
    routing, so the pattern the file shows is a router of its own: one
    categorical over the branch states, learned from every child's sentinel
    at once. Patterns seen on fewer than MIN_ROWS visible rows fold into one
    "other" state. Returns {parent: (codes for every row, -1 where unknown,
    number of states)}.
    """
    codes = {job[0]: job[1] for job in jobs}
    widths = {job[0]: job[2] for job in jobs}
    rows = np.flatnonzero(visible)
    children = {}
    for child, parent in parent_of.items():
        children.setdefault(parent, []).append(child)
    out = {}
    for parent, kids in children.items():
        if len(kids) < 3:
            continue
        groups = {}
        for child in kids:
            key = tuple(sorted(schema["items"][child]["gate"].get("observed_if", [])))
            groups.setdefault(key, []).append(child)
        pattern = np.zeros(len(rows), np.int64)
        for key in sorted(groups)[:10]:
            asked = np.mean([codes[child][rows] != widths[child] - 1 for child in groups[key]], axis=0)
            pattern = pattern * 2 + (asked >= 0.5)
        values, counts = np.unique(pattern, return_counts=True)
        keep = values[counts >= MIN_ROWS]
        if len(keep) < 2:
            continue
        lookup = {int(v): i for i, v in enumerate(keep)}
        other = len(keep)
        state = np.full(len(visible), -1, np.int64)
        state[rows] = [lookup.get(int(v), other) for v in pattern]
        n_states = other + 1 if (state[rows] == other).any() else other
        out[parent] = (state, n_states)
    return out


def _predict_states(ctx, state, n_states, visible, features, params):
    """The branch state's distribution for every row: out of fold on visible
    rows (two halves), the halves' average on held-out rows. Trees on the
    GIVEN codes plus the parent's distribution."""
    rows = np.flatnonzero(visible)
    p = np.zeros((len(visible), n_states))
    hidden = np.flatnonzero(~visible)
    grow = _lightgbm if lightgbm is not None else _histgb
    for half in (0, 1):
        held = rows[ctx.fold[rows] % 2 == half]
        fit = rows[ctx.fold[rows] % 2 != half]
        present, codes = np.unique(state[fit], return_inverse=True)
        target = np.concatenate([held, hidden])
        x_fit = np.hstack([ctx.z[fit].astype(float), features[fit]])
        x_out = np.hstack([ctx.z[target].astype(float), features[target]])
        q = np.zeros((len(target), n_states))
        if len(present) == 1:
            q[:, present[0]] = 1.0
        else:
            q[:, present] = grow(params, x_fit, codes, x_out, len(present),
                                 categorical=range(ctx.z.shape[1]))
        p[held] = q[:len(held)]
        p[hidden] += q[len(held):] / 2
    return p


def _sentinel_column(ctx, fit, y, width):
    """The GIVEN column most informative about whether the item was asked."""
    sentinel = (y[fit] == width - 1).astype(np.int64)
    if sentinel.sum() == 0 or sentinel.sum() == len(fit) or ctx.z.shape[1] == 0:
        return None
    gains = [_information(sentinel, ctx.z[fit, g], 2, ctx.cards[g]) for g in range(ctx.z.shape[1])]
    best = int(np.argmax(gains))
    return best if gains[best] > 0 else None


def _trees_parent(features, params):
    """Trees on the GIVEN codes plus the routers' predicted distributions."""
    def run(ctx, fit, out, y, width):
        present, codes = np.unique(y[fit], return_inverse=True)
        if len(present) == 1:
            return _onehot(np.full(len(out), present[0]), width)
        grow = _lightgbm if lightgbm is not None else _histgb
        x_fit = np.hstack([ctx.z[fit].astype(float), features[fit]])
        x_out = np.hstack([ctx.z[out].astype(float), features[out]])
        p = np.zeros((len(out), width))
        p[:, present] = grow(params, x_fit, codes, x_out, len(present),
                             categorical=range(ctx.z.shape[1]))
        return p
    run.label = "trees_parent"
    run.halves = True
    return run


def _routers(jobs, parent_of, visible):
    """Which scored item best says whether each gated child was asked.

    The schema names a parent for each gated item, but the file need not
    follow it (UNHCR's gates agree with the data at a median of 0.83, and two
    of its branches follow s3_12 rather than their declared parent s3_11). For
    every child with a scored parent, the mutual information between its
    sentinel and every small scored item's answer is counted on the visible
    rows. An item that beats the declared parent is the child's alternative
    router; the ROUTERS most-used alternatives are kept, so they can be
    predicted before the children that lean on them.

    Returns {child: [parent, alternative]} with the alternative left out
    where none was kept, and the set of all routers.
    """
    codes = {job[0]: job[1] for job in jobs}
    widths = {job[0]: job[2] for job in jobs}
    rows = np.flatnonzero(visible)
    small = [item for item in codes if widths[item] <= ROUTER_WIDTH]

    def descendants(item):
        found = set()
        for child, parent in parent_of.items():
            while parent in parent_of or parent == item:
                if parent == item:
                    found.add(child)
                    break
                parent = parent_of[parent]
        return found

    best = {}
    for child, parent in parent_of.items():
        sentinel = (codes[child][rows] == widths[child] - 1).astype(np.int64)
        if sentinel.sum() == 0 or sentinel.sum() == len(rows):
            continue
        below = descendants(child)
        gain = {}
        for item in small:
            if item == child or item in below:
                continue
            gain[item] = _information(sentinel, codes[item][rows], 2, widths[item])
        others = [(g, item) for item, g in gain.items() if item != parent]
        if parent not in gain or not others:
            continue
        other = max(others)
        if other[0] > gain[parent]:
            best[child] = other[1]
    used = {}
    for item in best.values():
        used[item] = used.get(item, 0) + 1
    parents = set(parent_of.values())
    kept = set(sorted((item for item in used if item not in parents),
                      key=lambda item: -used[item])[:ROUTERS]) | parents
    routers_of = {}
    for child, parent in parent_of.items():
        routers_of[child] = [parent] + ([best[child]] if best.get(child) in kept else [])
    return routers_of, kept


class _TaskHeads:
    """Vectorised loss and probabilities for unequal-width categorical heads.

    Padding belongs only to this temporary tensor, never to a returned vector.
    All items retain equal weight; ensemble members average their losses during
    training and their probabilities during inference, as in the scalar loop.
    """

    def __init__(self, widths, device):
        import torch
        self.widths = tuple(widths)
        self.width = max(widths)
        offsets = np.cumsum([0] + list(widths[:-1]))
        positions = np.arange(self.width)[None, :]
        valid = positions < np.asarray(widths)[:, None]
        indices = np.where(valid, offsets[:, None] + positions, 0)
        self.indices = torch.tensor(indices, dtype=torch.long, device=device)
        self.valid = torch.tensor(valid, dtype=torch.bool, device=device)

    def padded(self, logits):
        return logits[..., self.indices].masked_fill(~self.valid, float("-inf"))

    def losses(self, logits, targets):
        import torch
        padded = self.padded(logits)
        if logits.dim() == 3:
            targets = targets[:, None, :].expand(-1, logits.shape[1], -1)
        losses = torch.nn.functional.cross_entropy(
            padded.reshape(-1, self.width), targets.reshape(-1), reduction="none")
        return losses.reshape(targets.shape)

    def probabilities(self, logits):
        import torch
        p = torch.softmax(self.padded(logits), dim=-1)
        return p.mean(dim=1) if logits.dim() == 3 else p


def _multitask(ctx, ys, widths, visible, deadline, seed=SEED, refit=False,
               hidden=None, dropout=None, lr=None, members=1, honest=None):
    """One network for every scored item: a shared trunk over the GIVEN block
    and one softmax head per item, trained on all visible rows at once.

    Returns {index: (oof, final)}, both (rows, width). It trains on two folds
    (pairs of the stack's folds, so its predictions stay out of fold there);
    oof holds each half's prediction from the network trained on the other.
    final averages the two half networks. With `refit` a third network trained on every visible row for the
    number of epochs the halves found best joins that average, so the final
    prediction is both an ensemble and a model that saw all the data. Both
    phases use it now that the loss is vectorised. Empty if torch is
    missing or time is short; the stack then simply does without it.
    """
    try:
        import torch
    except Exception:
        return {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    hidden = NET_HIDDEN if hidden is None else hidden
    dropout = NET_DROPOUT if dropout is None else dropout
    lr = NET_LR if lr is None else lr
    x_all = torch.tensor(ctx.x, dtype=torch.float32, device=device)
    y_all = torch.tensor(np.stack(ys, axis=1), dtype=torch.long, device=device)
    offsets = np.concatenate([[0], np.cumsum(widths)])
    heads = _TaskHeads(widths, device)
    rows = np.flatnonzero(visible)
    honest = NET_HONEST_OOF if honest is None else honest

    class Members(torch.nn.Module):
        """TabM-mini (Gorishniy, Kotelnikov, Babenko, ICLR 2025): `members`
        ensemble members that share one trunk and differ by a per-member
        affine map of the input and their own output layer. Trained on the
        mean of the member losses; predicts the mean of the member
        probabilities. Returns logits of shape (rows, members, outputs)."""

        def __init__(self, d_in, d_out):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.randn(members, d_in).sign() + torch.randn(members, d_in) * 0.1)
            self.shift = torch.nn.Parameter(torch.zeros(members, d_in))
            self.trunk = torch.nn.Sequential(
                torch.nn.Linear(d_in, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout))
            self.heads = torch.nn.Parameter(torch.randn(members, hidden, d_out) / hidden ** 0.5)
            self.bias = torch.nn.Parameter(torch.zeros(members, d_out))

        def forward(self, x):
            views = x[:, None, :] * self.scale + self.shift            # (rows, members, d_in)
            h = self.trunk(views)
            return torch.einsum("rkh,kho->rko", h, self.heads) + self.bias

    def build():
        if members > 1:
            return Members(x_all.shape[1], int(offsets[-1])).to(device)
        return torch.nn.Sequential(
            torch.nn.Linear(x_all.shape[1], hidden), torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, int(offsets[-1]))).to(device)

    def loss_on(net, idx):
        return heads.losses(net(x_all[idx]), y_all[idx]).mean()

    def epoch(net, opt, fit_t, gen):
        net.train()
        order = fit_t[torch.randperm(len(fit_t), generator=gen).to(device)]
        for start in range(0, len(order), NET_BATCH):
            if time.time() > deadline:
                raise TimeoutError
            opt.zero_grad()
            loss_on(net, order[start:start + NET_BATCH]).backward()
            opt.step()
        if time.time() > deadline:
            raise TimeoutError

    def train(fit, held):
        """Each item's predictions from the epoch that was best for it.

        One network is trained; the epoch loop stops when the mean held-out
        loss stalls, as before, but every item keeps the predictions (for
        every row) from its own best epoch, since items with sharp
        structure peak later than the average item. Returns those
        snapshots, each item's best epoch and, with `honest`, fairer held
        predictions (else None).

        With `honest`, the held rows are split in two by fold, and each part's
        out-of-fold predictions come from the epoch that was best on the other
        part. Choosing the epoch on the rows it is then scored on made the
        network look better out of fold than it is, and the blend weighs
        candidates by exactly that score.
        """
        net = build()
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=NET_DECAY)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        fit_t = torch.tensor(fit, device=device)
        best_mean, stale = np.inf, 0
        best_loss = np.full(len(widths), np.inf)
        best_epoch = np.ones(len(widths), np.int64)
        snapshots = [None] * len(widths)
        held_pos = torch.tensor(held, device=device)
        part = ctx.fold[held] // 2            # held halves hold two stack folds each
        parts = [np.flatnonzero(part == k) for k in (0, 1)]
        split = honest and all(len(q) for q in parts)
        if split:
            part_pos = [torch.tensor(q, device=device) for q in parts]
            part_loss = [np.full(len(widths), np.inf) for _ in parts]
            # cross[k][j]: item j's predictions for the held rows of part k,
            # from the epoch that was best on the other part.
            cross = [[None] * len(widths) for _ in parts]
        for count in range(1, NET_EPOCHS + 1):
            epoch(net, opt, fit_t, gen)
            # One forward pass over every row serves both the held-out losses
            # and the snapshots, so the per-item epochs cost no extra pass.
            net.eval()
            with torch.no_grad():
                logits = net(x_all)
                per_item = heads.losses(logits[held_pos], y_all[held_pos])
                losses = per_item.mean(dim=tuple(range(per_item.dim() - 1))).double().cpu().numpy()
                if not NET_ITEM_EPOCHS:
                    losses = np.full(len(widths), float(losses.mean()))
                improved = losses < best_loss - 1e-4
                probs = None
                if improved.any():
                    # One device-to-host transfer, rather than one per item.
                    probs = heads.probabilities(logits).double().cpu().numpy()
                    for j in np.flatnonzero(improved):
                        best_loss[j], best_epoch[j] = losses[j], count
                        snapshots[j] = probs[:, j, :widths[j]].copy()
                if split:
                    for k in (0, 1):
                        here = per_item[part_pos[k]]
                        loss_k = here.mean(dim=tuple(range(here.dim() - 1))).double().cpu().numpy()
                        better = loss_k < part_loss[k] - 1e-4
                        if better.any():
                            if probs is None:
                                probs = heads.probabilities(logits).double().cpu().numpy()
                            other = held[parts[1 - k]]
                            for j in np.flatnonzero(better):
                                part_loss[k][j] = loss_k[j]
                                cross[1 - k][j] = probs[other, j, :widths[j]].copy()
            mean = float(losses.mean())
            if mean < best_mean - 1e-4:
                best_mean, stale = mean, 0
            else:
                stale += 1
                if stale >= NET_PATIENCE:
                    break
        if not split or any(c is None for side in cross for c in side):
            return snapshots, best_epoch, None
        held_p = []
        for j in range(len(widths)):
            p = np.zeros((len(held), widths[j]))
            for k in (0, 1):
                p[parts[k]] = cross[k][j]
            held_p.append(p)
        return snapshots, best_epoch, held_p

    def train_full(fit, epochs):
        """A network on every visible row for a fixed number of epochs."""
        net = build()
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=NET_DECAY)
        gen = torch.Generator(device="cpu").manual_seed(seed + 1000)
        fit_t = torch.tensor(fit, device=device)
        for _ in range(epochs):
            epoch(net, opt, fit_t, gen)
        return net

    def probabilities(net, idx, which=None):
        """Per-item probabilities for the rows idx; `which` limits the items."""
        net.eval()
        with torch.no_grad():
            logits = net(x_all[torch.tensor(idx, device=device)])
            probs = heads.probabilities(logits).double().cpu().numpy()
            return {j: probs[:, j, :widths[j]].copy()
                    for j in (range(len(widths)) if which is None else which)}

    oof = [np.zeros((len(visible), w)) for w in widths]
    final = [np.zeros((len(visible), w)) for w in widths]
    everyone = np.arange(len(visible))
    epochs = []
    for half in (0, 1):
        held = rows[ctx.fold[rows] % 2 == half]
        fit = rows[ctx.fold[rows] % 2 != half]
        if len(held) == 0 or len(fit) == 0:
            return {}
        snapshots, best_epoch, held_p = train(fit, held)
        epochs.append(int(np.median(best_epoch)))
        for j, p in enumerate(snapshots):
            oof[j][held] = p[held] if held_p is None else held_p[j]
            final[j] += p / 2
    if refit:
        try:
            net = train_full(rows, max(1, int(round(float(np.mean(epochs))))))
            full_probs = probabilities(net, everyone)
            final = [(final[j] + full_probs[j]) / 2 for j in range(len(widths))]
        except TimeoutError:
            pass                          # the half networks' average stands
    return {j: (oof[j], final[j]) for j in range(len(widths))}


def _em_step(w, likelihood, prior=0.0):
    """One EM update of the mixture weights, with `prior` pseudo-rows per
    candidate (the mode of a Dirichlet prior on the weights)."""
    share = w[:, None] * likelihood
    share /= share.sum(axis=0, keepdims=True)
    m, n = likelihood.shape
    return (share.sum(axis=1) + prior) / (n + m * prior)


def _mixture_weights(likelihood, rounds=30, start=None, prior=None):
    """EM for the weights of a mixture whose components are fixed.

    likelihood is (models, rows): each model's floored probability of the
    answer actually given. The mean log of the mixture is concave in the
    weights, so EM converges to the best mix, but slowly near the edge of the
    simplex. SQUAREM (Varadhan and Roland, 2008) extrapolates from two EM
    steps; a jump that scores worse than plain EM is discarded. Thirty rounds
    land closer to the optimum than 500 plain steps, in a quarter of the time.
    A Dirichlet prior of EM_PRIOR pseudo-rows per candidate keeps a weak
    candidate from taking weight by chance on a region with few rows.
    """
    likelihood = np.maximum(likelihood, FLOOR)
    m = likelihood.shape[0]
    prior = EM_PRIOR if prior is None else prior
    w = np.full(m, 1.0 / m) if start is None else np.asarray(start, float)

    def loss(v):
        return -(np.log(v @ likelihood).sum()
                 + prior * np.log(np.maximum(v, 1e-12)).sum()) / likelihood.shape[1]

    for _ in range(rounds):
        w1 = _em_step(w, likelihood, prior)
        w2 = _em_step(w1, likelihood, prior)
        r, v = w1 - w, w2 - 2 * w1 + w
        if np.linalg.norm(v) < 1e-12:
            w = w2
            continue
        a = min(-np.linalg.norm(r) / np.linalg.norm(v), -1.0)
        jump = np.maximum(w - 2 * a * r + a * a * v, 0.0)
        if jump.sum() > 0:
            jump = _em_step(jump / jump.sum(), likelihood, prior)
            if loss(jump) <= loss(w2):
                w2 = jump
        w = w2
    return w


def _fixed(oof, final, label="network"):
    """A candidate whose predictions were made elsewhere (the shared network).

    Rows it is asked about out of fold get their out-of-fold prediction; the
    final fit, on every visible row, gets the full-data prediction.
    """
    def run(ctx, fit, out, y, width, final_fit=False):
        p = (final if final_fit else oof)[out][:, :width]
        return p / p.sum(axis=1, keepdims=True)
    run.fixed = True
    run.label = label
    return run


def _mean_log(p, y):
    """Mean log score of the answers y, as the grader floors it."""
    p = _floored(p)
    return float(np.log(p[np.arange(len(y)), y]).mean())


def _linear_pool(probs, weights):
    mix = sum(w * p for w, p in zip(weights, probs))
    return mix / mix.sum(axis=1, keepdims=True)


def _log_pool(probs, weights):
    """Weighted geometric blend. The weights need not sum to one, so their
    scale also acts as a temperature."""
    logit = sum(w * np.log(_floored(p)) for w, p in zip(weights, probs))
    p = np.exp(logit - logit.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def _sharpen(p, tau):
    """Raise to the power tau and renormalise: above 1 sharper, below 1 flatter."""
    if tau == 1.0:
        return p
    p = np.exp(tau * np.log(np.maximum(p, 1e-12)))
    return p / p.sum(axis=1, keepdims=True)


def _log_pool_objective(w, logs, y):
    """Loss and gradient of the log pool under the grader's affine floor.

    Flooring the inputs makes logarithms finite, but is not the grader's
    final floor on the pooled distribution. Include that final floor in
    both the objective and its derivative, just as _mean_log does.
    """
    rows = np.arange(len(y))
    logit = logs @ w
    q = np.exp(logit - logit.max(axis=1, keepdims=True))
    q /= q.sum(axis=1, keepdims=True)
    scale = 1.0 - q.shape[1] * FLOOR
    true_q = q[rows, y]
    scored = FLOOR + scale * true_q
    loss = -np.log(scored).mean()
    attenuation = scale * true_q / scored
    grad = -(attenuation[:, None] * (
        logs[rows, y, :] - np.einsum("rk,rkm->rm", q, logs))).mean(axis=0)
    return loss, grad


def _log_pool_weights(oof, y, start):
    """Non-negative log-pool weights that maximise the grader's log score."""
    from scipy.optimize import minimize
    logs = np.stack([np.log(_floored(p.astype(float))) for p in oof], axis=2)
    start = np.asarray(start, float)
    fit = minimize(_log_pool_objective, start, args=(logs, y), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None)] * len(start), options={"maxiter": 100})
    # The floored objective is not convex; keep the starting pool if the
    # numerical optimiser fails to improve it.
    if (not np.isfinite(fit.x).all()
            or _log_pool_objective(fit.x, logs, y)[0] > _log_pool_objective(start, logs, y)[0]):
        return start
    return fit.x


def _strata_weights(picked, level, fold, w_global):
    """Blend weights per level of one GIVEN column, shrunk toward the global
    weights (hierarchical stacking, Yao et al. 2022, with a fixed pull).

    picked is (models, rows) likelihoods, level the column's code per row.
    Each half of the rows (by fold parity) gets weights fitted on the other
    half, so the returned score is honest. Returns (score, table) where
    table maps a level to its weights, fitted on every row.
    """
    levels = np.unique(level)
    rows = np.arange(len(level))
    mixed = np.zeros(len(level))

    def fit_table(mask):
        table = {}
        for value in levels:
            here = mask & (level == value)
            n = int(here.sum())
            if n == 0:
                table[value] = w_global
                continue
            w = _mixture_weights(picked[:, here], start=w_global)
            table[value] = (n * w + STRATA_SHRINK * w_global) / (n + STRATA_SHRINK)
        return table

    for half in (0, 1):
        held = fold % 2 == half
        if not held.any() or held.all():
            return None
        table = fit_table(~held)
        for value in levels:
            here = held & (level == value)
            mixed[here] = table[value] @ picked[:, here]
    score = float(np.log(np.maximum(mixed, FLOOR)).mean())
    return score, fit_table(np.ones(len(level), bool))


def _meta_features(probs):
    """Every candidate's log-probabilities side by side, floored as the grader floors."""
    return np.hstack([np.log(_floored(p.astype(float))) for p in probs])


def _meta_learner(oof, y, fold):
    """A multinomial logistic regression over all candidates' out-of-fold
    log-probabilities (stacking; Dirichlet calibration is its one-candidate
    case). Scored by an inner two-fold split so its number is honest, unlike
    the recipe choices, which are made on the rows they are scored on.
    Returns (inner score, inner predictions, model fitted on every row), or
    None when a fold lacks a class.
    """
    x = _meta_features(oof)
    width = oof[0].shape[1]
    pred = np.zeros((len(y), width))
    for half in (0, 1):
        held = fold % 2 == half
        if not held.any() or held.all() or len(np.unique(y[~held])) < 2:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(C=META_C, max_iter=MAX_ITER)
            model.fit(x[~held], y[~held])
        p = np.zeros((int(held.sum()), width))
        p[:, model.classes_] = model.predict_proba(x[held])
        pred[held] = p
    inner = _mean_log(pred, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(C=META_C, max_iter=MAX_ITER)
        model.fit(x, y)
    return inner, pred, model


def _label_of(candidate):
    return getattr(candidate, "label", candidate.__name__.strip("_"))


def _held_out_loss(likelihood, folds):
    """Log loss, summed over rows, of the mix of `likelihood` (models, rows)
    when each fold's rows are scored with weights fitted on the other folds."""
    total = 0.0
    for k in np.unique(folds):
        held = folds == k
        weights = _mixture_weights(likelihood[:, ~held])
        total -= float(np.log(np.maximum(weights @ likelihood[:, held], FLOOR)).sum())
    return total


class _Stack:
    """Out-of-fold scores and final predictions per candidate, for one region.

    Candidates can be added in more than one pass; the mixture weights are
    refitted over everything scored so far each time.
    """

    def __init__(self, ctx, fit, out, y, width, scale=None, sentinel_rate=0.0,
                 original_out=None, total_width=None):
        self.ctx, self.fit, self.out, self.y, self.width = ctx, fit, out, y, width
        self.total_width = width if total_width is None else total_width   # options the grader sees
        self.original_out = out if original_out is None else original_out
        self.scale = scale       # indices of the item's ordered levels, or None
        self.sentinel_rate = sentinel_rate   # the sentinel's share in a region modelled without it
        self.picked, self.finals, self.labels, self.failed = [], [], [], []
        self.seconds = {}        # label -> seconds spent, for SB_DEBUG
        self.oof = []            # RAW normalised OOF predictions, like self.finals
        self.kind = "linear"     # how the last result() blended, for SB_DEBUG
        self.errors = {}         # first error text per failed candidate, for SB_DEBUG
        self.weights = None      # EM weights for self.picked, refitted after each add
        self.choice = None       # (candidates, recipe, unfloor) from the last choose()
        self.unfloor_found, self.unfloor_gain = False, None   # set by _choose, for SB_DEBUG
        self.lock = threading.RLock()   # the TabICL thread and a pass worker may add at once
        self.fit_loss = None     # out-of-fold log loss of that mix, summed over rows
        self.full_context = set()   # labels whose held-out prediction came from a fit on every fit row

    def _halves(self, candidate, deadline):
        """Two folds; the final prediction averages the two fold models, or,
        for a candidate marked `full_final` (in-context learners in Test,
        where the clock allows a third fit), comes from one fit on every
        fit row: more context is worth 0.003 to 0.005 nats on the proxy."""
        ctx, fit, y, width = self.ctx, self.fit, self.y, self.width
        query_out = (self.original_out if getattr(candidate, "preserve_query_batch", False)
                     else self.out)
        # Regions and their filtered query rows retain sorted respondent order.
        select = np.searchsorted(query_out, self.out)
        oof = np.full((len(fit), width), 1.0 / width)
        finals = []
        for half in (0, 1):
            held = ctx.fold[fit] % 2 == half
            if not held.any() or held.all():
                continue
            if time.time() > deadline:
                raise TimeoutError
            p = candidate(ctx, fit[~held], np.concatenate([fit[held], query_out]), y, width)
            raw = p[:held.sum()]
            oof[held] = raw / raw.sum(axis=1, keepdims=True)
            finals.append(p[held.sum():][select])
        if len(finals) < 2:
            raise ValueError("too few rows for two folds")
        if len(self.out) and getattr(candidate, "full_final", False) and (_FULL_FINAL.get("on") or getattr(candidate, "always_full_final", False)):
            try:
                if time.time() > deadline:
                    raise TimeoutError
                return oof, candidate(ctx, fit, query_out, y, width)[select], True
            except TimeoutError:
                pass
        return oof, np.mean(finals, axis=0), False

    def add(self, candidates, deadline):
        ctx, fit, y, width = self.ctx, self.fit, self.y, self.width
        folds = ctx.fold[fit]
        for candidate in candidates:
            label = _label_of(candidate)
            if label in self.labels:
                continue                  # already in this stack from an earlier pass
            oof = np.full((len(fit), width), 1.0 / width)
            full = False
            began = time.time()
            try:
                if getattr(candidate, "halves", False):
                    oof, final, full = self._halves(candidate, deadline)
                else:
                    for k in range(FOLDS):
                        held = folds == k
                        if not held.any() or held.all():
                            continue
                        if time.time() > deadline:
                            raise TimeoutError
                        train, test = fit[~held], fit[held]
                        raw = candidate(ctx, train, test, y, width)
                        oof[held] = raw / raw.sum(axis=1, keepdims=True)
                    if not len(self.out):
                        final = np.empty((0, width))
                    elif time.time() > deadline:
                        raise TimeoutError         # the final fit costs as much as a fold
                    elif getattr(candidate, "fixed", False):
                        final = candidate(ctx, fit, self.out, y, width, final_fit=True)
                    else:
                        final = candidate(ctx, fit, self.out, y, width)
            except Exception as exc:
                # A candidate that fails anywhere is out; SB_DEBUG counts it.
                with self.lock:
                    if isinstance(exc, TimeoutError):
                        self.failed.append("timeout:" + label)
                    else:
                        self.failed.append("failed:" + label)
                        self.errors.setdefault(label, repr(exc))
                continue
            with self.lock:
                self.seconds[label] = self.seconds.get(label, 0.0) + time.time() - began
                # EM consumes scored likelihoods; all other calibration paths
                # consume raw vectors and apply their scoring floor exactly once.
                self.picked.append(_floored(oof)[np.arange(len(fit)), y[fit]])
                self.oof.append(oof.astype(np.float32))
                self.finals.append(final / final.sum(axis=1, keepdims=True))
                self.labels.append(label)
                if full:
                    self.full_context.add(label)
        with self.lock:
            self._refit()

    def _refit(self):
        """Fit the mix once per add, here in the worker, from the last weights.

        Sorting items and assembling them then reuse it; refitting there, on
        the main thread, cost tens of seconds per pass on UNHCR.
        """
        if not self.picked or (self.weights is not None
                               and len(self.weights) == len(self.picked)):
            return
        picked = np.array(self.picked)
        start = None
        if self.weights is not None and len(self.weights) < len(picked):
            new = len(picked) - len(self.weights)
            start = np.concatenate([0.9 * self.weights, np.full(new, 0.1 / new)])
        self.weights = _mixture_weights(picked, start=start)
        self.fit_loss = -float(np.log(np.maximum(self.weights @ picked, FLOOR)).sum())

    def loss(self):
        """Out-of-fold log loss of the current mix, summed over rows."""
        with self.lock:
            self._refit()
            return self.fit_loss

    def gain_of(self, labels, before):
        """Held-out gain in nats, summed over rows, of adding the candidates
        named in `labels` to the mix of those in `before` (labels present
        earlier), or None when there is nothing to compare.

        Only the named candidates are compared, so one another thread adds
        meanwhile earns them nothing. The mix weights are fitted on the other
        folds' rows, so a candidate is not credited for fitting the rows it is
        scored on.
        """
        with self.lock:
            index = {label: i for i, label in enumerate(self.labels)}
            old = [index[label] for label in before if label in index]
            new = [index[label] for label in labels if label in index and label not in before]
            if not old or not new:
                return None
            picked = np.array(self.picked)
        folds = self.ctx.fold[self.fit]
        if len(np.unique(folds)) < 2:
            return None
        return _held_out_loss(picked[old], folds) - _held_out_loss(picked[old + new], folds)

    def recipe(self):
        """How to blend: ("linear" or "log", weights, temperature, smoothing).

        The plain linear blend unless choose() ran after the last add.
        """
        with self.lock:
            self._refit()
            if self.choice is not None and self.choice[0] == len(self.picked):
                return self.choice[1]
            return "linear", self.weights, 1.0, 0.0

    def unfloor(self):
        """Whether choose() found the floor reply better out of fold for this blend."""
        with self.lock:
            return self.choice is not None and self.choice[0] == len(self.picked) and self.choice[2]

    def choose(self, meta=False, strata=False):
        """Pick the blend recipe; run at the end, as it costs per item. The
        cheap choices (temperature, log pool) come first for every region;
        the dear ones (weights per level, meta-learner) in a second pass."""
        with self.lock:
            self._refit()
            if self.picked:
                self.unfloor_found = False
                recipe = self._choose(meta, strata)
                self.choice = (len(self.picked), recipe, self.unfloor_found)

    def _choose(self, meta=False, strata=False):
        """The linear blend with EM weights is the default. With enough rows, a
        temperature on that blend and a geometric (log) blend are also tried,
        and whichever scores best out of fold is used. A log blend can be sure
        where its sharpest model is sure; a linear one never is sharper than
        its parts, which costs on the many near-certain UNHCR cells. With
        `meta` and enough rows, the stacking meta-learner is tried too, and
        wins only if its inner-fold score beats the best recipe.
        """
        weights = self.weights
        if not BLEND_CHOICE or len(self.fit) < POOL_MIN_ROWS or len(self.oof) < 2:
            return "linear", weights, 1.0, 0.0
        y = self.y[self.fit]
        linear = _linear_pool(self.oof, weights)
        best, best_score, mixed = ("linear", weights, 1.0), _mean_log(linear, y), linear
        for tau in TEMPERATURES:
            sharpened = _sharpen(linear, tau)
            score = _mean_log(sharpened, y)
            if score > best_score:
                best, best_score, mixed = ("linear", weights, tau), score, sharpened
        try:
            log_weights = _log_pool_weights(self.oof, y, weights)
            pooled = _log_pool(self.oof, log_weights)
            score = _mean_log(pooled, y)
            if score > best_score:
                best, best_score, mixed = ("log", log_weights, 1.0), score, pooled
        except Exception:
            pass
        # Weights per level of one GIVEN column (survey round, country, arm),
        # for the column whose honest score is best; only a clear gain over
        # the plain linear blend's own score counts, as that one is in-sample.
        if strata and len(self.fit) >= META_MIN_ROWS and self.ctx.z.shape[1] > 0:
            try:
                picked = np.array(self.picked)
                fold = self.ctx.fold[self.fit]
                plain = _mean_log(linear, y)
                found = None
                columns = [g for g in np.argsort(self.ctx.cards, kind="stable")
                           if self.ctx.cards[g] <= STRATA_MAX + 1][:STRATA_COLUMNS]
                for g in columns:
                    got = _strata_weights(picked, self.ctx.z[self.fit, g], fold, weights)
                    if got is not None and got[0] > plain + 1e-4 and (found is None or got[0] > found[0]):
                        found = (got[0], g, got[1])
                if found is not None:
                    table = found[2]
                    strata_mix = np.zeros_like(linear)
                    level = self.ctx.z[self.fit, found[1]]
                    for value, w in table.items():
                        here = level == value
                        strata_mix[here] = _linear_pool([o[here] for o in self.oof], w)
                    for tau in (1.0,) + tuple(TEMPERATURES):
                        score = _mean_log(_sharpen(strata_mix, tau), y)
                        if score > best_score:
                            best, best_score, mixed = ("strata", (found[1], table, weights), tau), score, _sharpen(strata_mix, tau)
            except Exception:
                pass
        if meta and len(self.fit) >= META_MIN_ROWS and LogisticRegression is not None:
            try:
                got = _meta_learner(self.oof, y, self.ctx.fold[self.fit])
                if got is not None and got[0] > best_score + 1e-4:
                    best, best_score, mixed = ("meta", got[2], 1.0), got[0], got[1]
            except Exception:
                pass
        # Last, on an ordered scale, a little mass to neighbouring levels.
        lam_best = 0.0
        if self.scale is not None and (self.scale < self.width).all():
            for lam in SMOOTHING:
                score = _mean_log(_smooth(mixed, self.scale, lam), y)
                if score > best_score:
                    lam_best, best_score = lam, score
        # Last, send the floor's best reply instead of the blend itself, when
        # that scores better out of fold over all the options the grader sees.
        if UNFLOOR and self.sentinel_rate == 0:
            if lam_best > 0:
                mixed = _smooth(mixed, self.scale, lam_best)
            rows = np.arange(len(y))
            plain = mixed / mixed.sum(axis=1, keepdims=True)
            plain = FLOOR + (1.0 - self.total_width * FLOOR) * plain[rows, y]
            scored = _unfloored(mixed, self.total_width)[1][rows, y]
            self.unfloor_gain = float(np.log(scored).mean() - np.log(plain).mean())
            self.unfloor_found = self.unfloor_gain > UNFLOOR_MARGIN
        return best + (lam_best,)

    def result(self):
        """The blended prediction for the target rows.

        If the chosen blend fails or gives a non-finite value, the plain
        linear blend is used instead.
        """
        with self.lock:
            return self._result()

    def _tilted(self, weights):
        """The blend weights to use on the held-out rows.

        The weights are fitted on out-of-fold predictions. On held-out rows
        of ten real surveys the blend scored better with more than the
        fitted weight on the three candidates that carry it: TabPFN-3.5,
        TabICL and the first network (about +0.0005 nats per answer in all,
        over three rounds of replays). The cause is not established; this
        is a measured correction. The total is kept: in a log pool it acts
        as a temperature.
        """
        weights = np.asarray(weights, dtype=float)
        factor = np.array([TILT.get(label, 1.0) if label not in TILT_NEEDS_FULL_FIT or label in self.full_context
                           else 1.0 for label in self.labels])
        if len(factor) != len(weights) or (factor == 1.0).all():
            return weights
        tilted = weights * factor
        return tilted * (weights.sum() / tilted.sum()) if tilted.sum() > 0 else weights

    def _result(self):
        try:
            kind, weights, tau, lam = self.recipe()
            if kind == "meta":
                p = np.zeros((len(self.out), self.width))
                p[:, weights.classes_] = weights.predict_proba(_meta_features(self.finals))
            elif kind == "strata":
                column, table, w_global = weights
                level = self.ctx.z[self.out, column]
                p = _linear_pool(self.finals, self._tilted(w_global))
                for value, w in table.items():
                    here = level == value
                    if here.any():
                        p[here] = _linear_pool([f[here] for f in self.finals], self._tilted(w))
                p = _sharpen(p, tau)
            else:
                pool = _log_pool if kind == "log" else _linear_pool
                p = _sharpen(pool(self.finals, self._tilted(weights)), tau)
            if lam > 0:
                p = _smooth(p, self.scale, lam)
            if self.unfloor():
                p = _unfloored(p, self.total_width)[0]
            if np.isfinite(p).all():
                self.kind = (kind + ("" if tau == 1.0 else "+temperature")
                             + ("" if lam == 0 else "+smooth")
                             + ("+unfloor" if self.unfloor() else ""))
                return p
        except Exception:
            pass
        self.kind = "linear-fallback"
        return _linear_pool(self.finals, self.weights)


def _gate_state(frame, schema, item, given_set, cache):
    """(closed, open) row masks implied by GIVEN answers alone.

    closed: the gate is shut for sure, so the item was never asked.
    open:   the gate is open for sure. Only decided when the parent is GIVEN;
            a scored parent leaves the item open to doubt.
    """
    if item in cache:
        return cache[item]
    n = len(frame)
    gate = schema["items"][item].get("gate")
    if not gate:
        state = (np.zeros(n, bool), np.ones(n, bool))
    else:
        parent, allowed = gate["parent"], set(gate["observed_if"])
        if parent in given_set:
            hit = frame[parent].isin(allowed).to_numpy()
            state = (~hit, hit)
        elif parent in schema["items"] and schema["items"][parent].get("class") == "PREDICT":
            closed, _ = _gate_state(frame, schema, parent, given_set, cache)
            state = (closed.copy(), np.zeros(n, bool))
        else:
            state = (np.zeros(n, bool), np.zeros(n, bool))
    cache[item] = state
    return state


def _parts(y, width, gated, closed, opened, visible, marginal):
    """Split one item's rows into what the gate decides and what has to be modelled.

    Returns the item's starting (rows, width) prediction, a list of
    (fit rows, target rows, modelled width) regions, and the vector to send
    on rows whose gate a GIVEN answer shuts (None when there is no such rule).
    """
    n = len(y)
    out = np.tile(marginal, (n, 1))
    sentinel = width - 1 if gated else None

    # Trust the gate where nearly every visible respondent agrees with it: a
    # closed gate gives the visible closed rows' own distribution (the sentinel,
    # or almost), an open gate models the other options and gives the sentinel
    # the small share seen on the visible open rows.
    closed_rate = open_rate = None
    if sentinel is not None and closed[visible].any():
        closed_rate = float((y[visible & closed] == sentinel).mean())
    if sentinel is not None and opened[visible].any():
        open_rate = float((y[visible & opened] == sentinel).mean())
    rule_closed = (closed_rate is not None and closed_rate >= RULE_AGREEMENT
                   and (visible & closed).sum() >= RULE_MIN_ROWS)
    rule_open = (open_rate is not None and open_rate <= 1.0 - RULE_AGREEMENT
                 and (visible & opened).sum() >= RULE_MIN_ROWS)

    region = np.ones(n, bool)
    reply = None
    if rule_closed:
        belief = _smoothed(np.bincount(y[visible & closed], minlength=width).astype(float))
        out[closed] = belief
        if UNFLOOR:
            reply = _unfloored(belief[None, :], width)[0][0]   # the grader then scores the belief itself
        region &= ~closed
    split = []
    if rule_open:
        split.append((opened & region, width - 1, open_rate))   # the sentinel is (nearly) impossible
        region &= ~opened
    split.append((region, width, 0.0))

    parts = []
    for rows, w, rate in split:
        fit = np.flatnonzero(visible & rows)
        target = np.flatnonzero(rows)
        if w < width:
            fit = fit[y[fit] != sentinel]     # the few sentinel rows stay out of the model
        if len(target) == 0 or len(fit) < MIN_ROWS or len(np.unique(y[fit])) < 2:
            if len(target) and len(fit):
                out[target] = _smoothed(np.bincount(y[fit], minlength=width).astype(float))
            continue
        parts.append((fit, target, w, rate))
    return out, parts, reply


def _one_thread():
    """Hold OpenMP to one thread in the calling worker.

    The limit is per thread, so one set in the main thread does not reach the
    workers; without it each tree fit there starts a thread per core and a
    pass runs several times slower.
    """
    if threadpool_limits is None:
        return contextlib.nullcontext()
    return threadpool_limits(1, user_api="openmp")


def _report(stacks, started, marks, budget):
    """Which candidates ran, what weight they earned and when each pass ended.

    Only with SB_DEBUG set to a file path; the line is appended there because
    the grader keeps a successful run's output to itself.
    """
    weight, count, failed, kinds, errors, seconds = {}, {}, {}, {}, {}, {}
    for _, regions in stacks.values():
        for stack in regions:
            kinds[stack.kind] = kinds.get(stack.kind, 0) + 1
            for label in stack.failed:
                failed[label] = failed.get(label, 0) + 1
            for label, text in stack.errors.items():
                errors.setdefault(label, text)
            for label, spent in stack.seconds.items():
                seconds[label] = seconds.get(label, 0.0) + spent
            if not stack.picked:
                continue
            for label, w in zip(stack.labels, stack.weights):
                weight[label] = weight.get(label, 0.0) + w
                count[label] = count.get(label, 0) + 1
    line = ("%.1fs budget=%.0fs icl=%s kumo=%s " % (time.time() - started, budget, _ICL.get("ok"),
                                                     _KUMO.get("ok"))
            + " ".join("%s:n=%d,w=%.3f,s=%.2f" % (k, count[k], weight[k] / count[k],
                                              seconds.get(k, 0.0) / count[k])
                       for k in sorted(count))
            + "".join(" %s=%d" % (k, failed[k]) for k in sorted(failed))
            + " blend:" + ",".join("%s=%d" % (k, kinds[k]) for k in sorted(kinds))
            + " passes:" + ",".join("%s@%.0fs" % mark for mark in marks)
            + "".join(" error[%s]=%s" % (k, errors[k][:300].replace("\n", " "))
                      for k in sorted(errors)))
    with open(os.environ["SB_DEBUG"], "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _phase_two(schema, frame, visible):
    """Phase 2 shows TRAIN and DEV with answers, and the frame holds every respondent."""
    split = schema.get("split") or {}
    n_rows = (schema.get("dataset") or {}).get("n_rows")
    if split.get("n_dev") and int(visible.sum()) >= split.get("n_train", 0) + split["n_dev"]:
        return True
    return bool(n_rows) and len(frame) >= n_rows


def _need(items):
    return min(BUDGET_MAX, BUDGET + BUDGET_PER_ITEM * items)


def _budget(phase_two, items, now):
    """Seconds for this instrument: its share of the clock still left.

    The instruments not called yet are taken from KNOWN_ITEMS, so a lone run
    of one instrument (score.py runs each in its own process) still keeps
    time for the other two, and on the platform the last instrument gets
    whatever the earlier ones left.
    """
    total = (TOTAL_TEST if phase_two else TOTAL_DEV) * SCALE
    left = total - RESERVE * SCALE - (now - _IMPORTED)
    pending = list(KNOWN_ITEMS)
    for count in _CALLED + [items]:
        if count in pending:
            pending.remove(count)
    pending = sorted(pending, reverse=True)[:max(0, 2 - len(_CALLED))]
    share = _need(items) / (_need(items) + sum(_need(count) for count in pending))
    _CALLED.append(items)
    # The fast candidates need some seconds even when the clock is nearly gone,
    # but never more than half the reserve past the shared limit.
    floor = max(5.0 * SCALE, min(30.0 * SCALE, left + RESERVE * SCALE / 2))
    return max(floor, left * share)


def predict(frame, schema):
    started = time.time()
    rng = np.random.default_rng(SEED)

    items = [name for name, record in schema["items"].items()
             if record["class"] in ("GIVEN", "PREDICT")]
    given = [name for name in items if schema["items"][name]["class"] == "GIVEN"]
    given_set = set(given)
    options = {item: _options(schema, item) for item in items}

    grid = frame[items]
    targets = [name for name in items if schema["items"][name]["class"] == "PREDICT"]
    # A failed run may be retried; a weaker successful one is final on Test.
    real = REQUIRE_MODELS and USE_FOUNDATION and len(targets) in KNOWN_ITEMS
    if real:
        if not _icl_ready():
            raise RuntimeError("TabICL did not load")
        if USE_KUMO and not _kumo_ready():
            raise RuntimeError("TabPFN-3.5 did not load")
        _models_work()
    visible = ~grid[targets].isna().any(axis=1).to_numpy() if targets else np.ones(len(frame), bool)

    phase_two = _phase_two(schema, frame, visible)
    budget = _budget(phase_two, len(targets), started)
    deadline = started + budget
    _ICL["estimators"] = ICL_ESTIMATORS_TEST if phase_two else ICL_ESTIMATORS

    # The crowd marginal for every item: the answer for anything we cannot model.
    marginal = {}
    for item in items:
        counts = np.bincount(np.maximum(_codes(frame[item], options[item]), 0)
                             [frame[item].notna().to_numpy()],
                             minlength=len(options[item])).astype(float)
        marginal[item] = _smoothed(counts)

    full = {}
    shut_reply = {}          # item -> (rows a GIVEN answer shuts, the vector to send there)
    stacks = {}
    marks = []                    # (pass, seconds) for SB_DEBUG
    try:
        ctx = _Context(frame, schema, given, rng)
        gate_cache = {}
        jobs = []
        for item in targets:
            y = _codes(frame[item], options[item])
            if (y[visible] < 0).any():
                continue                  # visible answer outside the schema
            closed, opened = _gate_state(frame, schema, item, given_set, gate_cache)
            jobs.append((item, y, len(options[item]),
                         bool(schema["items"][item].get("gate")), closed, opened))

        # Nearest neighbours by GIVEN profile; seconds. Only the household
        # vote's fallback reads them, so only where there is a household key.
        if ctx.household is not None:
            try:
                ctx.neighbours(visible, started + 0.1 * budget)
            except Exception:
                ctx.near = ctx.far = None

        # One network over every item first; it owns the CPU or GPU while it
        # trains. Seeds share the folds, so their average stays out of fold.
        shared, nets = {}, []
        if jobs and visible.sum() >= MIN_ROWS:
            for seed in NET_SEEDS:
                began = time.time()
                try:
                    net = _multitask(ctx, [job[1] for job in jobs], [job[2] for job in jobs],
                                     visible, started + NET_SHARE * budget, seed,
                                     refit=True)
                except Exception:
                    break
                if not net:
                    break
                nets.append(net)
                if time.time() + (time.time() - began) > started + NET_SHARE * budget:
                    break                 # the next seed would not finish
        marks.append(("net%d" % len(nets), time.time() - started))
        try:
            if nets:
                shared = {jobs[j][0]: (np.mean([net[j][0] for net in nets], axis=0),
                                       np.mean([net[j][1] for net in nets], axis=0))
                          for j in nets[0]}
        except Exception:
            shared = {}
        # Test: a second architecture as its own candidate, so the blend picks
        # between the two on the real data instead of one being chosen blind.
        shared_b = {}
        if phase_two and nets:
            nets_b = []
            for seed in NET_SEEDS_B:
                began = time.time()
                try:
                    net = _multitask(ctx, [job[1] for job in jobs], [job[2] for job in jobs],
                                     visible, started + NET_SHARE_TEST * budget, seed,
                                     refit=True, **NET_B)
                except Exception:
                    break
                if not net:
                    break
                nets_b.append(net)
                if time.time() + (time.time() - began) > started + NET_SHARE_TEST * budget:
                    break
            try:
                if nets_b:
                    shared_b = {jobs[j][0]: (np.mean([net[j][0] for net in nets_b], axis=0),
                                             np.mean([net[j][1] for net in nets_b], axis=0))
                                for j in nets_b[0]}
            except Exception:
                shared_b = {}
            marks.append(("netb%d" % len(nets_b), time.time() - started))
        # Test: TabM-mini, a third estimator family, as its own candidate.
        shared_c = {}
        if phase_two and nets and time.time() < started + NET_SHARE_TEST * budget:
            try:
                net = _multitask(ctx, [job[1] for job in jobs], [job[2] for job in jobs],
                                 visible, started + NET_SHARE_TEST * budget, NET_SEEDS_C[0],
                                 refit=True, **NET_C)
                if net:
                    shared_c = {jobs[j][0]: net[j] for j in net}
            except Exception:
                shared_c = {}
            marks.append(("netc%d" % (1 if shared_c else 0), time.time() - started))

        # Cross-round household linkage (v92a): seconds; off where nothing links.
        panel = None
        if PANEL and jobs and shared:
            try:
                panel = _panel_build(frame, schema, given, ctx, jobs, visible)
            except Exception as exc:
                panel = None
                _panel_note("panel error " + repr(exc)[:300])
        marks.append(("panel%d" % (panel is not None), time.time() - started))
        # Fast candidates are numpy only (milliseconds per item); medium ones
        # take a fraction of a second; dear ones up to seconds.
        fast = [_marginal] + [_backoff(c, a) for c, a in BACKOFF]
        # No neighbour-kernel candidates from v85: the blend scores better on held-out rows without them.
        if ctx.near is not None and ctx.household is not None:
            fast.append(_twins_household)
        trees = []
        if (lightgbm is not None or HistGradientBoostingClassifier is not None) and ctx.z.shape[1] > 0:
            # LightGBM is fast enough for four folds; sklearn's trees get two.
            trees.append(_trees(TREES, "trees", halves=lightgbm is None))
        medium = [_backoff(c, a, greedy=True, discount=d) for c, a, d in LADDERS]
        regression = [_regression(LR_C)] if LogisticRegression is not None and not LEAN else []
        # Trees are the one lever with a measured Dev effect, so every variant
        # reaches every item before anything judged only on invented data.
        trees_all = [_trees(TREES_DEEP, "trees_deep", halves=True)] if trees and not LEAN else []
        if trees and lightgbm is not None and not LEAN:
            trees_all.append(_trees_encoded(TREES_ENCODED, "trees_encoded"))
        deep = []
        if trees and lightgbm is not None and not LEAN:
            deep.append(_trees(TREES_WIDE, "trees_wide", halves=True))
        if trees and phase_two and not LEAN:
            deep.append(_trees(TREES_TUNED, "trees_tuned", halves=True))
        dear = []
        if LogisticRegression is not None and ctx.z.shape[1] > 1 and not LEAN:
            dear.append(_regression_pairs(LR_C, PAIRS, PAIR_SHRINK))
        icl = []
        if _icl_ready() and not KUMO_ONLY:
            _ICL["until"] = started + ICL_SHARE * budget
            icl.append(_icl)
        kumo = []
        if _kumo_ready():
            _KUMO["estimators"] = KUMO_ESTIMATORS_TEST if phase_two else KUMO_ESTIMATORS
            _KUMO["until"] = started + ICL_SHARE * budget
            kumo.append(_kumo)
        # Extra members in Test only, TabPFN-3.5 first: it carries the most blend weight.
        _ICL["more_until"] = _KUMO["more_until"] = started + MORE_SHARE * budget
        more = (([_kumo_more] if kumo else []) + ([_icl_more] if icl else [])) if phase_two else []
        spare = {}           # {"on": True} once the machine has shown it is fast enough for them
        pfn = []
        pfn_share = PFN_SHARE_TEST if phase_two else PFN_SHARE
        if not LEAN and _pfn_ready():
            _PFN["estimators"] = PFN_ESTIMATORS_TEST if phase_two else PFN_ESTIMATORS
            _PFN["until"] = started + pfn_share * budget
            pfn.append(_pfn)
        _FULL_FINAL["on"] = phase_two
        narrow = lambda stack: pfn if stack.width <= PFN_MAX_CLASSES else []

        stacks = {}

        def first(job):
            try:
                with _one_thread():
                    return (job[0],) + start_regions(job)
            except Exception:
                return job[0], None, []

        def start_regions(job):
            item, y, width, gated, closed, opened = job
            out, parts, reply = _parts(y, width, gated, closed, opened, visible, marginal[item])
            if reply is not None and np.isfinite(reply).all() and reply.sum() > 0:
                shut_reply[item] = (closed, reply)
            regions = []
            scale = _scale(options[item], item)
            for fit, target, w, rate in parts:
                # Visible fit rows get OOF predictions in distribution(); their
                # full-fit predictions are never needed. In two-fold candidates
                # they also duplicated the held rows in the inference request.
                original_out = target
                target = target[~np.isin(target, fit)]
                stack = _Stack(ctx, fit, target, y, w, scale=scale, sentinel_rate=rate,
                               original_out=original_out, total_width=width)
                extra = [_fixed(*shared[item])] if item in shared else []
                if item in shared_b:
                    extra.append(_fixed(*shared_b[item], label="network_b"))
                if item in shared_c:
                    extra.append(_fixed(*shared_c[item], label="network_c"))
                if panel is not None and item in shared and w == width and item in panel["index"]:
                    extra += _panel_candidates(panel, item, *shared[item])
                stack.add(extra + fast, deadline)
                regions.append(stack)
            return out, regions

        def adder(candidates, share, extra=None, floor=0.0):
            """Adds candidates to one item, unless `share` of the budget is gone;
            `extra(stack)` can add candidates built for that region. A `floor`
            keeps the pass open for at least that share of the budget from now,
            so a slow earlier pass cannot squeeze it out."""
            until = min(deadline, max(started + share * budget, time.time() + floor * budget))

            def run(item):
                try:
                    with _one_thread():
                        for stack in stacks[item][1]:
                            if time.time() < until:
                                stack.add(candidates + (extra(stack) if extra else []), deadline)
                except Exception:
                    pass
                return item
            return run

        def ordinal_for(stack):
            """The ordered-levels candidate, for regions whose item is a scale."""
            if trees and stack.scale is not None and (stack.scale < stack.width).all():
                return [_ordinal(TREES, stack.scale)]
            return []

        # Children of a scored parent: the parent's predicted distribution
        # feeds a parent-mixture candidate and parent-aware trees.
        parent_of = {}
        for item, *_ in jobs:
            gate = schema["items"][item].get("gate")
            if gate and schema["items"].get(gate["parent"], {}).get("class") == "PREDICT":
                parent_of[item] = gate["parent"]
        codes_of = {job[0]: job[1] for job in jobs}

        def depth_of(item):
            depth = 0
            while item in parent_of:
                item, depth = parent_of[item], depth + 1
            return depth

        def distribution(parent):
            """The parent's predicted distribution for every row: out of fold
            on visible rows, the current blend on held-out rows. An open region
            modelled without its sentinel gets the sentinel's small share here
            too, as assemble() gives it, so children see the same thing on
            visible and held-out rows."""
            out, regions = stacks[parent]
            p = out.copy()
            for stack in regions:
                if stack.picked:
                    q = _linear_pool([o.astype(float) for o in stack.oof], stack.weights)
                    p[stack.fit] = 0.0
                    p[stack.fit, :stack.width] = q * (1.0 - stack.sentinel_rate)
                    if stack.sentinel_rate > 0:
                        p[stack.fit, stack.width] = stack.sentinel_rate
            return p

        try:
            routers_of, routers = _routers(jobs, parent_of, visible)
        except Exception:
            routers_of = {item: [parent] for item, parent in parent_of.items()}
            routers = set(parent_of.values())
        try:
            branch_of = _branch_codes(jobs, parent_of, visible, schema)
        except Exception:
            branch_of = {}
        branch_p = {}

        def gate_adder(parents):
            def run(item):
                try:
                    parent = parent_of[item]
                    if parent not in parents or time.time() > started + GATE_SHARE * budget:
                        return item
                    found = [r for r in routers_of.get(item, [parent]) if r in parents]
                    features = np.hstack([parents[r] for r in found])
                    candidates = [_parent_mix(parents[parent], codes_of[parent])]
                    if trees:
                        candidates.append(_trees_parent(features, TREES_PARENT))
                    with _one_thread():
                        for stack in stacks[item][1]:
                            extra = []
                            if trees and stack.width == len(options[item]) and stack.sentinel_rate == 0:
                                shared_trees = {}
                                extra = [_chain(parents[parent], codes_of[parent], TREES_PARENT,
                                                features, cache=shared_trees)]
                                column = _sentinel_column(ctx, stack.fit, stack.y, stack.width)
                                if column is not None:
                                    extra.append(_chain(parents[parent], codes_of[parent], TREES_PARENT,
                                                        features, label="chain_given", given=column,
                                                        cache=shared_trees))
                                for other in found[1:]:
                                    extra.append(_chain(parents[other], codes_of[other], TREES_PARENT,
                                                        features, label="chain_router", cache=shared_trees))
                                if parent in branch_p:
                                    extra.append(_chain(branch_p[parent], branch_of[parent][0], TREES_PARENT,
                                                        features, label="chain_branch", cache=shared_trees))
                            stack.add(candidates + extra, deadline)
                except Exception:
                    pass
                return item
            return run

        def chooser(item):
            """The cheap blend choices of one item, unless time is up."""
            try:
                for stack in stacks[item][1]:
                    if time.time() < deadline:
                        stack.choose()
            except Exception:
                pass
            return item

        def chooser_dear(item):
            """The dear blend choices (weights per level, meta-learner), while
            their share of the budget lasts."""
            try:
                for stack in stacks[item][1]:
                    if time.time() < started + META_SHARE * budget:
                        stack.choose(meta=True, strata=True)
            except Exception:
                pass
            return item

        def worst_first():
            """Routers first (the items they route inherit them), then by loss."""
            try:
                return sorted(stacks, key=lambda item: (item not in routers, -sum(
                    s.loss() for s in stacks[item][1] if s.picked)))
            except Exception:
                return list(stacks)

        def assemble(item):
            """The item's prediction; a region that fails keeps the marginal,
            and never stops the other items."""
            out, regions = stacks[item]
            for stack in regions:
                if not stack.picked:
                    continue
                try:
                    p = stack.result()
                    if p.shape == (len(stack.out), stack.width) and np.isfinite(p).all():
                        out[stack.out] = 0.0
                        out[stack.out, :stack.width] = p * (1.0 - stack.sentinel_rate)
                        if stack.sentinel_rate > 0:
                            out[stack.out, stack.width] = stack.sentinel_rate
                except Exception:
                    pass
            return out

        # Threads, not processes: the heavy loops in sklearn release the GIL,
        # and nothing has to be pickled. BLAS is held to one thread here and
        # OpenMP inside each worker (_one_thread), so eight workers do not
        # each spin up eight more.
        limit = (threadpool_limits(1) if threadpool_limits is not None else None)
        try:
            with ThreadPoolExecutor(max_workers=JOBS) as pool:
                # Pass 1: every item gets the fast candidates and the network.
                for item, out, regions in pool.map(first, jobs):
                    if out is not None:
                        stacks[item] = (out, regions)
                        full[item] = assemble(item)
                marks.append(("fast", time.time() - started))
                # TabICL in the background from here, routers first, while the
                # CPU passes run; the TabICL pass below only fills what it missed.
                icl_thread = None
                if (icl or kumo) and ICL_CONCURRENT and time.time() < started + ICL_SHARE * budget:
                    icl_order = worst_first()

                    def icl_background():
                        # TabICL on every item first; Kumo-Tabular, slower per
                        # item, only in what is left, so it never costs TabICL items.
                        def run(candidates, share):
                            for item in icl_order:
                                if time.time() > started + share * budget:
                                    return False
                                try:
                                    with _one_thread():
                                        for stack in stacks[item][1]:
                                            if time.time() < started + share * budget:
                                                stack.add(candidates, deadline)
                                except Exception:
                                    pass
                            return True

                        done = [run(candidates, ICL_SHARE) for candidates in (icl, kumo)]
                        # The extra-member candidates follow only on a machine fast
                        # enough to have finished both passes this early.
                        if more and all(done) and time.time() < started + FAST_SHARE * budget:
                            spare["on"] = True
                            for extra in more:
                                run([extra], MORE_SHARE)

                    icl_thread = threading.Thread(target=icl_background, daemon=True)
                    icl_thread.start()
                # Routers pass: the gate parents and the other routers get every
                # candidate now, so the children's chain and router candidates
                # inherit them.
                parents_first = [item for item in worst_first() if item in routers]
                if parents_first and time.time() < started + PARENTS_SHARE * budget:
                    everything = trees + deep + medium + regression    # TabICL and the pair regression wait for their passes
                    for item in pool.map(adder(everything, PARENTS_SHARE), parents_first):
                        full[item] = assemble(item)
                    marks.append(("parents", time.time() - started))
                # Trees on every item before anything dearer, with their own
                # window: the proven lever is never starved by the passes before it.
                if trees:
                    for item in pool.map(adder(trees, TREES_SHARE, floor=TREES_FLOOR), worst_first()):
                        full[item] = assemble(item)
                    marks.append(("trees", time.time() - started))
                # Pilot: keep only the optional families that pay on these rows.
                optional = [("trees2", trees_all, ordinal_for), ("deep", deep, None),
                            ("regression", regression, None), ("dear", dear, None), ("pfn", [], narrow)]
                optional = [f for f in optional if f[1] or (f[0] == "pfn" and pfn)]
                keep = {name for name, _, _ in optional}
                if PILOT and optional and time.time() < started + TREES2_SHARE * budget:
                    order = worst_first()
                    sample = order[::max(1, len(order) // PILOT_ITEMS)][:PILOT_ITEMS]
                    found = {}
                    for name, candidates, extra in optional:
                        if time.time() > started + TREES2_SHARE * budget:
                            break
                        cutoff = min(deadline, time.time() + PILOT_FAMILY_SHARE * budget)

                        def trial(item, candidates=candidates, extra=extra, cutoff=cutoff):
                            # Regions where the family was already present or failed add no rows.
                            gain = rows = 0.0
                            try:
                                with _one_thread():
                                    for stack in stacks[item][1]:
                                        if stack.picked:
                                            family = candidates + (extra(stack) if extra else [])
                                            with stack.lock:
                                                before = set(stack.labels)
                                            stack.add(family, cutoff)
                                            got = stack.gain_of([_label_of(c) for c in family], before)
                                            if got is not None:
                                                gain += got
                                                rows += len(stack.fit)
                            except Exception:
                                pass
                            return gain, rows
                        began = time.time()
                        results = list(pool.map(trial, sample))
                        rows = sum(r for _, r in results)
                        found[name] = (sum(g for g, _ in results) / max(rows, 1.0), rows, time.time() - began)
                    keep = {name for name in keep if name not in found or found[name][0] > PILOT_MIN_GAIN}
                    for item in sample:
                        full[item] = assemble(item)
                    marks.append(("pilot:" + ",".join("%s=%.4f/%dr/%.0fs" % (n, g, r, s) for n, (g, r, s) in found.items()),
                                  time.time() - started))
                trees_all = trees_all if "trees2" in keep else []
                deep = deep if "deep" in keep else []
                regression = regression if "regression" in keep else []
                dear = dear if "dear" in keep else []
                pfn = pfn if "pfn" in keep else []
                medium = medium + regression
                # Deep and target-encoded trees on every item, with the ordinal
                # candidate for scale items, before the unmeasured passes.
                if trees_all and time.time() < started + TREES2_SHARE * budget:
                    for item in pool.map(adder(trees_all, TREES2_SHARE, ordinal_for), worst_first()):
                        full[item] = assemble(item)
                    marks.append(("trees2", time.time() - started))
                # TabICL next, routers first, so the gate pass inherits it;
                # then TabPFN v2 in what is left of the same window.
                if icl_thread is not None:
                    icl_thread.join(max(0.0, started + ICL_SHARE * budget - time.time()) + 5.0)
                    marks.append(("iclbg", time.time() - started))
                if icl and time.time() < started + ICL_SHARE * budget:
                    for item in pool.map(adder(icl, ICL_SHARE), worst_first()):
                        full[item] = assemble(item)
                    marks.append(("icl", time.time() - started))
                if kumo and time.time() < started + ICL_SHARE * budget:
                    for item in pool.map(adder(kumo, ICL_SHARE), worst_first()):
                        full[item] = assemble(item)
                    marks.append(("kumo", time.time() - started))
                if more and spare.get("on") and time.time() < started + MORE_SHARE * budget:
                    for item in pool.map(adder(more, MORE_SHARE), worst_first()):
                        full[item] = assemble(item)
                    marks.append(("more", time.time() - started))
                if pfn and time.time() < started + pfn_share * budget:
                    # Only regions inside TabPFN's class limit get it; wider ones skip it cleanly.
                    for item in pool.map(adder([], pfn_share, narrow), worst_first()):
                        full[item] = assemble(item)
                    marks.append(("pfn", time.time() - started))
                # Gate pass: children of scored parents, shallowest first, so
                # a parent's distribution is settled before its children use it.
                levels = {}
                for item, parent in parent_of.items():
                    if USE_GATE_CANDIDATES and item in stacks and parent in stacks:
                        levels.setdefault(depth_of(item), []).append(item)
                for depth in sorted(levels):
                    if time.time() > started + GATE_SHARE * budget:
                        break
                    parents = {}
                    for item in levels[depth]:
                        for router in routers_of.get(item, [parent_of[item]]):
                            if router not in parents and router in stacks:
                                try:
                                    full[router] = assemble(router)   # held-out rows get its latest blend too
                                    parents[router] = distribution(router)
                                except Exception:
                                    pass
                    # The branch state of each family at this level, predicted once.
                    for item in levels[depth]:
                        parent = parent_of[item]
                        if parent in branch_of and parent not in branch_p and parent in parents and trees:
                            try:
                                state, n_states = branch_of[parent]
                                branch_p[parent] = _predict_states(ctx, state, n_states, visible,
                                                                   parents[parent], TREES_PARENT)
                            except Exception:
                                pass
                    for item in pool.map(gate_adder(parents), levels[depth]):
                        full[item] = assemble(item)
                if levels:
                    marks.append(("gate", time.time() - started))
                # Later passes: deep trees, regression and ladders, then the
                # pair regression, worst-fitted items first, while time lasts.
                for name, candidates, share, extra in (("deep", deep, DEEP_SHARE, None),
                                                       ("medium", medium, MEDIUM_SHARE, None),
                                                       ("dear", dear, DEAR_SHARE, None)):
                    if not candidates or time.time() > started + share * budget:
                        continue
                    for item in pool.map(adder(candidates, share, extra), worst_first()):
                        full[item] = assemble(item)
                    marks.append((name, time.time() - started))
                # Last, each item picks its blend recipe while time lasts,
                # worst-fitted items first, as they have most to gain.
                if BLEND_CHOICE:
                    for item in pool.map(chooser, worst_first()):
                        full[item] = assemble(item)
                    marks.append(("blend", time.time() - started))
                    if time.time() < started + META_SHARE * budget:
                        for item in pool.map(chooser_dear, worst_first()):
                            full[item] = assemble(item)
                        marks.append(("blend2", time.time() - started))
        finally:
            if limit is not None:
                limit.unregister()
        if icl_thread is not None and icl_thread.is_alive():
            marks.append(("iclbg-alive", time.time() - started))   # still in one fit; it stops at its next check
        if os.environ.get("SB_DEBUG"):
            _report(stacks, started, marks, budget)
    except Exception:
        if real:
            raise                 # a failed run may be retried; crowd marginals would be final
        # Never fatal: every item not yet assembled keeps its crowd marginal.
        # On practice machines (SB_DEBUG set) the error is logged; the
        # platform never sets SB_DEBUG, so nothing is written there.
        if os.environ.get("SB_DEBUG"):
            import traceback
            with open(os.environ["SB_DEBUG"], "a", encoding="utf-8") as fh:
                fh.write("ERROR after %d items: %s\n"
                         % (len(full), traceback.format_exc().replace("\n", " | ")))

    if real:
        _require_success(stacks, phase_two)

    # One vector per blank PREDICT cell, rows top to bottom, items in schema
    # order. GIVEN columns are never scored, so a blank there gets nothing.
    values = grid.to_numpy(dtype=object)
    columns = [items.index(item) for item in targets]
    vectors = []
    for row in range(values.shape[0]):
        for column in columns:
            item = items[column]
            if pd.isna(values[row, column]):
                vector = full[item][row] if item in full else marginal[item]
                if item in full and item in shut_reply and shut_reply[item][0][row]:
                    vector = shut_reply[item][1]
                vector = np.asarray(vector, dtype=float)
                if not np.all(np.isfinite(vector)) or vector.sum() <= 0:
                    vector = marginal[item]
                vectors.append(vector.tolist())
    return vectors
