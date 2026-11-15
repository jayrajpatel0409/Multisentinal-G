"""CampaignSim: a generative simulator of an Indian multilingual e-commerce review
marketplace containing (i) honest users, (ii) hard-negative honest behaviour
(families, flash-sale crowds, terse generic reviewers, AI-assisted honest writers) and
(iii) coordinated paid-review campaigns with tunable adversarial evasion.

Everything is generated from scratch with a seeded RNG.  The simulator emits
  * a review table (user, product, time, rating, text-embedding, image-hash, ...)
  * upstream detector scores (text-AI, image-mismatch, sentiment-gap) drawn from
    binormal models with a configurable AUC per language  -> stands in for Stages 1-3
  * an account table with ground truth.
"""
from dataclasses import dataclass, field, replace
import numpy as np
import pandas as pd
from scipy.stats import norm

LANGS = ["hi", "ta", "te", "bn", "mr", "gu"]


@dataclass
class SimConfig:
    n_genuine: int = 4000
    n_products: int = 250
    n_cat: int = 15
    T: float = 365.0
    dim: int = 24
    lang_share: tuple = (.30, .18, .14, .14, .12, .12)
    # honest hard negatives
    terse_frac: float = 0.30        # users who write generic short reviews
    n_phrases: int = 6              # generic phrases per language
    ai_assist_frac: float = 0.05    # honest reviews polished with an LLM
    family_frac: float = 0.04       # honest accounts in households / friend groups
    flash_frac: float = 0.05        # products with an honest flash-sale crowd
    sale_share: float = 0.35        # share of honest reviews written inside seasonal sale windows
    n_sales: int = 4
    sale_len: float = 7.0
    # campaigns
    prevalence: float = 0.08        # fake accounts / all accounts
    cell_size: tuple = (6, 18)
    n_targets: tuple = (3, 6)
    target_q: float = 0.7           # prob. a worker reviews a given target
    attack_frac: float = 0.3        # share of campaigns that 1-star a competitor
    sigma_text: float = 0.60        # paraphrase spread (higher = less duplicate-like)
    n_templates: int = 2
    win_scale: float = 3.0          # days: temporal spread of a campaign burst
    camo_mean: float = 0.0          # extra camouflage reviews per worker beyond matching an honest activity level
    catalog_share: float = 0.03     # share of honest image reviews that reuse the product's catalogue photo
    camo_hub: bool = False          # camouflage on top-popularity products inside sale windows
    img_prob: float = 0.5
    img_pool: int = 6
    img_reuse: float = 0.7
    new_acct_frac: float = 0.3      # share of workers on freshly created accounts (rest: aged / bought accounts)
    llm_frac: float = 0.6           # share of campaign reviews written by an LLM
    camp_sale: float = 0.25         # prob. a campaign burst is scheduled inside a sale window (timing camouflage)
    # upstream detectors (stand-ins for Stage 1-3), AUC per language order LANGS
    auc_text: tuple = (.92, .85, .83, .88, .85, .82)
    text_drop: float = 0.0          # fraction of separation lost to paraphrase evasion
    auc_img: float = 0.85
    obs_frac: float = 1.0           # fraction of reviews visible to the analyst (crawl coverage)


def _unit(rng, n, d):
    v = rng.standard_normal((n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _norm_rows(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def simulate(cfg: SimConfig, seed: int):
    rng = np.random.default_rng(seed)
    d, T = cfg.dim, cfg.T
    n_g, n_p = cfg.n_genuine, cfg.n_products
    n_fake_target = int(round(cfg.prevalence / (1 - cfg.prevalence) * n_g)) if cfg.prevalence > 0 else 0

    cat_vec = _unit(rng, cfg.n_cat, d)
    lang_vec = _unit(rng, len(LANGS), d)
    phrase_vec = _unit(rng, len(LANGS) * cfg.n_phrases, d).reshape(len(LANGS), cfg.n_phrases, d)

    # ---- products
    pop = 1.0 / np.arange(1, n_p + 1) ** 1.0
    rng.shuffle(pop)
    pop /= pop.sum()
    p_cat = rng.integers(0, cfg.n_cat, n_p)
    p_launch = rng.uniform(0, 0.8 * T, n_p)
    p_quality = np.clip(rng.normal(3.9, 0.5, n_p), 1.5, 4.9)
    cat_products = [np.where(p_cat == c)[0] for c in range(cfg.n_cat)]
    sale_start = rng.uniform(20, T - cfg.sale_len - 1, cfg.n_sales)
    top_pop = np.argsort(-pop)[:20]

    def honest_time(p):
        if rng.random() < cfg.sale_share:
            t = rng.choice(sale_start) + rng.uniform(0, cfg.sale_len)
            if t >= p_launch[p]: return t
        t = p_launch[p] + rng.exponential(60.0)
        return t if t <= T else rng.uniform(p_launch[p], T)

    # ---- genuine users
    g_lang = rng.choice(len(LANGS), n_g, p=np.array(cfg.lang_share) / sum(cfg.lang_share))
    g_fav = np.stack([rng.choice(cfg.n_cat, 2, replace=False) for _ in range(n_g)])
    g_terse = rng.random(n_g) < cfg.terse_frac
    g_n = 1 + rng.negative_binomial(2, 0.4, n_g)
    g_age = np.exp(rng.normal(np.log(400), 0.9, n_g))

    rows = []  # dict rows -> DataFrame

    def honest_text(u_lang, cat, terse):
        if terse:
            return _norm_rows((phrase_vec[u_lang, rng.integers(cfg.n_phrases)] + 0.08 * rng.standard_normal(d) / np.sqrt(d))[None])[0]
        noise = rng.standard_normal(d); noise /= np.linalg.norm(noise)
        return _norm_rows((0.5 * cat_vec[cat] + 0.3 * lang_vec[u_lang] + noise)[None])[0]

    def add_honest(u, p, t, extra=None):
        terse = bool(g_terse[u])
        emb = honest_text(g_lang[u], p_cat[p], terse)
        rating = int(np.clip(round(p_quality[p] + rng.normal(0, 1.0)), 1, 5))
        has_img = rng.random() < 0.25
        if has_img:
            h = 10_000_000 + p if rng.random() < cfg.catalog_share else None
            h = h if h is not None else -2  # -2 -> unique id assigned later
            mism = rng.random() < 0.03
        else:
            h, mism = -1, False
        llm = (not terse) and rng.random() < cfg.ai_assist_frac
        rows.append(dict(user=u, product=p, t=min(t, T), rating=rating, emb=emb, img=h, mism=mism,
                         llm=llm, fake=False, lang=g_lang[u], gap_fake=0, camo=False))

    for u in range(n_g):
        for _ in range(g_n[u]):
            if rng.random() < 0.7:
                c = rng.choice(g_fav[u]); cand = cat_products[c]
                p = rng.choice(cand, p=pop[cand] / pop[cand].sum()) if len(cand) else rng.choice(n_p, p=pop)
            else:
                p = rng.choice(n_p, p=pop)
            add_honest(u, p, honest_time(p))

    # honest flash-sale crowds
    flash_products = rng.choice(n_p, int(cfg.flash_frac * n_p), replace=False)
    flash_set = set()
    for p in flash_products:
        m = rng.integers(15, 41)
        users = rng.choice(n_g, m, replace=False)
        t0 = rng.uniform(0.2 * T, T - 3)
        for u in users:
            flash_set.add(int(u)); add_honest(u, p, t0 + rng.exponential(1.0))

    # households / friend groups
    group_of = {}
    fam_users = rng.permutation(n_g)[: int(cfg.family_frac * n_g)]
    i, fid = 0, 0
    while i < len(fam_users):
        sz = rng.integers(3, 7); members = fam_users[i:i + sz]; i += sz
        if len(members) < 3: break
        c = rng.choice(g_fav[members[0]]); cand = cat_products[c]
        prods = rng.choice(cand, min(len(cand), rng.integers(2, 5)), replace=False)
        for p in prods:
            t0 = rng.uniform(p_launch[p], T - 3)
            for u in members:
                add_honest(u, p, t0 + rng.exponential(1.0))
        for u in members: group_of[int(u)] = f"fam{fid}"
        fid += 1

    # ---- coordinated campaigns
    fake_rows_start = len(rows)
    users_meta = {}
    next_uid = n_g
    cell_id = 0
    cell_members = {}
    mu_text = [np.sqrt(2) * norm.ppf(a) for a in cfg.auc_text]
    while next_uid - n_g < n_fake_target:
        sz = int(rng.integers(cfg.cell_size[0], cfg.cell_size[1] + 1))
        sz = min(sz, n_fake_target - (next_uid - n_g)) if n_fake_target - (next_uid - n_g) >= 3 else sz
        mem = list(range(next_uid, next_uid + sz)); next_uid += sz
        nt = int(rng.integers(cfg.n_targets[0], cfg.n_targets[1] + 1))
        w = pop ** 0.3; w /= w.sum()
        targets = rng.choice(n_p, nt, replace=False, p=w)
        attack = rng.random() < cfg.attack_frac
        start = (rng.choice(sale_start) + rng.uniform(0, cfg.sale_len / 2)) if rng.random() < cfg.camp_sale else rng.uniform(0, T - 30)
        cl = int(rng.integers(len(LANGS)))
        cents = _unit(rng, cfg.n_templates, d)
        pool = [20_000_000 + cell_id * 1000 + k for k in range(cfg.img_pool)]
        for u in mem:
            ul = cl if rng.random() < 0.8 else int(rng.integers(len(LANGS)))
            age = rng.exponential(40.0) if rng.random() < cfg.new_acct_frac else np.exp(rng.normal(np.log(400), 0.9))
            users_meta[u] = (ul, age, f"cell{cell_id}")
            nt_rev = 0
            for p in targets:
                if rng.random() > cfg.target_q: continue
                nt_rev += 1
                noise = rng.standard_normal(d); noise /= np.linalg.norm(noise)
                emb = _norm_rows((cents[rng.integers(cfg.n_templates)] + cfg.sigma_text * noise)[None])[0]
                has_img = rng.random() < cfg.img_prob
                if has_img:
                    img = int(rng.choice(pool)) if rng.random() < cfg.img_reuse else -2
                    mism = rng.random() < 0.5
                else:
                    img, mism = -1, False
                rows.append(dict(user=u, product=int(p), t=min(start + rng.exponential(cfg.win_scale), T),
                                 rating=(1 if rng.random() < 0.65 else 2) if attack else (5 if rng.random() < 0.65 else 4), emb=emb, img=img, mism=mism,
                                 llm=rng.random() < cfg.llm_frac, fake=True, lang=ul, gap_fake=1, camo=False))
            # camouflage reviews
            honest_total = 1 + rng.negative_binomial(2, 0.4)            # workers mimic an honest activity level
            for _ in range(max(0, honest_total - nt_rev) + rng.poisson(cfg.camo_mean)):
                if cfg.camo_hub:
                    p = int(rng.choice(top_pop)); t = rng.choice(sale_start) + rng.uniform(0, cfg.sale_len)
                    if t < p_launch[p]: t = honest_time(p)
                else:
                    p = int(rng.choice(n_p, p=pop)); t = honest_time(p)
                noise = rng.standard_normal(d); noise /= np.linalg.norm(noise)
                emb = _norm_rows((0.5 * cat_vec[p_cat[p]] + 0.3 * lang_vec[ul] + noise)[None])[0]
                rows.append(dict(user=u, product=p, t=min(t, T), rating=int(np.clip(round(p_quality[p] + rng.normal(0, 1)), 1, 5)),
                                 emb=emb, img=-1, mism=False, llm=rng.random() < 0.5 * cfg.llm_frac,
                                 fake=True, lang=ul, gap_fake=1, camo=True))
        cell_members[f"cell{cell_id}"] = mem
        cell_id += 1

    R = pd.DataFrame(rows)
    emb = np.stack(R.pop("emb").values).astype(np.float32)
    # unique image ids for "-2"
    m = (R.img == -2).values
    R.loc[m, "img"] = 30_000_000 + np.arange(m.sum())
    R["review_id"] = np.arange(len(R))

    # ---- upstream detector scores (stand-ins for Stages 1-3)
    mu = np.array(mu_text)[R.lang.values]
    sep = mu * np.where(R.fake & R.llm, 1 - cfg.text_drop, 1.0)
    R["s_text"] = rng.standard_normal(len(R)) + sep * R.llm.values
    mu_img = np.sqrt(2) * norm.ppf(cfg.auc_img)
    R["has_img"] = R.img >= 0
    R["s_img"] = np.where(R.has_img, rng.standard_normal(len(R)) + mu_img * R.mism.values, 0.0)
    R["s_gap"] = np.abs(rng.normal(0, 0.5 + 0.25 * R.gap_fake.values))

    # ---- account table
    n_acc = next_uid
    A = pd.DataFrame({"user": np.arange(n_acc)})
    A["fake"] = A.user >= n_g
    lang = np.zeros(n_acc, int); age = np.zeros(n_acc); grp = np.empty(n_acc, object)
    lang[:n_g] = g_lang; age[:n_g] = g_age
    for u in range(n_g): grp[u] = group_of.get(u, f"solo{u}")
    for u, (ul, ag, gid) in users_meta.items():
        lang[u] = ul; age[u] = ag; grp[u] = gid
    A["lang"] = lang; A["age"] = age; A["group"] = grp
    A["cell"] = [g if g.startswith("cell") else "" for g in grp]
    A["terse"] = np.r_[g_terse, np.zeros(n_acc - n_g, bool)]
    A["family"] = [str(g).startswith("fam") for g in grp]
    A["flash"] = [int(u) in flash_set for u in range(n_acc)]
    ai_users = set(R.loc[(~R.fake) & R.llm, "user"].unique().tolist())
    A["ai_assist"] = [int(u) in ai_users for u in range(n_acc)]

    # ---- partial observation
    if cfg.obs_frac < 1.0:
        keep = rng.random(len(R)) < cfg.obs_frac
        R, emb = R[keep].reset_index(drop=True), emb[keep]
    R["rid"] = np.arange(len(R))
    return R, emb, A, dict(cell_members=cell_members, T=T, n_products=n_p)
