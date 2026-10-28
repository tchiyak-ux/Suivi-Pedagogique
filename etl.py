"""Harmonisation, jointure et statistiques des 4 fichiers d'indicateurs."""
import re, unicodedata, itertools
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan


def nrm(s):
    s = unicodedata.normalize("NFKD", str(s).lower().strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


# Matières : formes (normalisées) -> libellé canonique (FR/EN fusionnés)
_ALIAS = {
    "Geography": "geographie|geography",
    "History": "histoire|history",
    "French": "francais|french",
    "English": "anglais|english language",
    "Philosophy": "philosophie|philosophy",
    "Spanish": "espagnol", "German": "allemand", "Italian": "italien", "Chinese": "chinois",
    "Citizenship Education": "ecm|citizenship education",
    "Physical Education": "education physique et sportive|physical education and sports",
    "Computer Science / ICT": "informatique|computer science|ict|informatique tic|informatique fondamentale",
    "Mathematics": "mathematiques|mathematics",
    "Guidance and Counselling": "orientation conseil|guidance and counselling",
    "Psychopedagogy": "psychopedagogie|psycho pedagogy",
    "Didactics of Mathematics": "didactique des mathematiques|didactics of mathematics",
    "Didactics of French": "didactique du francais|didactics of french",
    "Didactics of Sciences and Technologies": "didactique des sciences et technologies|didactics of sciences and technologies",
    "Didactics of Social Studies": "didactique des sciences humaines et sociales|didactics of human and social sciences|didactics of social studies",
    "Didactics of Arts": "didactics of arts|didactics of arts and craft education",
    "Didactics of PE": "didactics of physical education and sports|didactics of physical education and sports education",
    "Human Rights Education": "human rights education and education in emergency situations|education for human rights and emergency situations|human rights and education in emergency situations",
    "Professional Ethics": "professional ethics and deontology applied to education|deontology and professional ethics applied to education|deontology and educational ethics",
    "Education Policies and Sociology": "sociologie et politiques de l education|sociology and policies of education|education policies and sociology of education|sociology and educational policies",
    "Inclusive Education": "inclusive education|education inclusive",
    "Educational Research": "educational research|applied research in education|applied pedagogic research|applied educational research|recherche appliquee a l education",
    "Educational Statistics": "educational statistics|statistics applied to education",
    "School Legislation": "school legislation and administration|legislation|professional legislation",
    "Pattern Drafting": "pattern drafting|pattern making",
}
MAP = {a: k for k, v in _ALIAS.items() for a in v.split("|")}


def matiere(s):
    n = re.sub(r"\s+", " ", re.sub(r"\b(ok|to be deleted)\b", "", nrm(s))).strip()
    return MAP.get(n, n.title())


_TAIL = {"all", "a", "c", "d", "ab", "ac", "abi", "ti"}


def niveau(c):
    t = nrm(c).replace("classe de ", "").split()
    while len(t) > 1 and t[-1] in _TAIL:
        t.pop()
    return " ".join(t).capitalize()


def cycle(n):
    if not isinstance(n, str):
        return np.nan
    n = n.lower()
    if n.startswith("year"): return "Formation normale"
    if any(k in n for k in ("sixth", "seconde", "premiere", "terminale")): return "Second cycle"
    if "annee" in n: return "Cycle technique (années)"
    return "Premier cycle"


REN = {"Département": "Departement", "Établissement": "Etablissement", "Sous-système": "Sous_systeme",
       "Type d'éducation": "Type_education", "Matière": "Matiere"}


def prep(df):
    d = df.rename(columns=lambda c: REN.get(c.strip(), c.strip())).copy()
    d["Etablissement"] = d["Etablissement"].astype(str).str.strip().str.upper()
    d["eid"] = d["Etablissement"].map(nrm)
    d["Sous_systeme"] = np.where(d["Sous_systeme"].astype(str).str.lower().str.contains("franco"), "Francophone", "Anglophone")
    t = d["Type_education"].astype(str).str.lower()
    d["Type_education"] = np.select([t.str.contains("techn"), t.str.contains("teacher|normal")], ["Technique", "Normal"], "Général")
    d["Arrondissement"] = d["Arrondissement"].astype(str).str.strip()
    d["Matiere_brute"] = d["Matiere"]
    d["Matiere"] = d["Matiere"].map(matiere)
    if "Classe" in d:
        d["Niveau"] = d["Classe"].map(niveau)
        d["Cycle"] = d["Niveau"].map(cycle)
    return d


def detect(df):
    c = {x.strip() for x in df.columns}
    if "Heures hebdo réelles" in c: return "assiduite"
    if {"Prévu", "Réalisé"} <= c: return "couverture"
    if "Total réussi" in c or "Taux de reusite" in c: return "reussite"
    if "Classe" in c: return "manque"


def clean(kind, df, cap=True, seuil=150):
    d = prep(df)
    if kind == "assiduite":
        d["prev"] = d["Horaires hebdo prévus"]; d["reel"] = d["Heures hebdo réelles"]
        d["aberrant"] = d.reel / d.prev * 100 >= seuil
        d["reel_adj"] = np.minimum(d.reel, d.prev) if cap else d.reel
        d["taux"] = d.reel_adj / d.prev * 100
    elif kind == "couverture":
        d["Prevu"] = d["Prévu"]; d["Realise"] = d["Réalisé"]
        d["taux"] = np.where(d.Prevu > 0, d.Realise / d.Prevu.replace(0, np.nan) * 100, np.nan)
    elif kind == "reussite":
        d["Effectif"] = d["Total"]; d["Filles"] = d["Nombre de filles"]
        d["Garcons"] = d["Nombre de garçons"].fillna(d.Effectif - d.Filles)
        d["Reussis"] = d["Total réussi"].fillna(d["Nombre de garçons réussis"] + d["Nombre de filles réussies"])
        d["Filles_reussies"] = d["Nombre de filles réussies"]
        d["Garcons_reussis"] = d["Nombre de garçons réussis"].fillna(d.Reussis - d.Filles_reussies)
        d["taux"] = d.Reussis / d.Effectif * 100
    else:
        d["n"] = 1
    return d


def valid(kind, d):
    return {"couverture": lambda x: x[x.Prevu > 0], "assiduite": lambda x: x[~x.aberrant],
            "reussite": lambda x: x[x.taux.notna()], "manque": lambda x: x}[kind](d)


CFG = {"couverture": ("Realise", "Prevu"), "assiduite": ("reel_adj", "prev"), "reussite": ("Reussis", "Effectif")}


def aggregate(kind, d, keys):
    k = [x for x in keys if x in d.columns]
    v = valid(kind, d)
    if kind == "couverture":
        g = v.groupby(k).agg(Prevu=("Prevu", "sum"), Realise=("Realise", "sum"), Nb_enseignants=("Enseignant", "nunique")).reset_index()
        g["Couverture"] = g.Realise / g.Prevu * 100
    elif kind == "assiduite":
        g = v.groupby(k).agg(Heures_prevues=("prev", "sum"), Heures_reelles=("reel_adj", "sum"), Nb_ens_assiduite=("Enseignant", "nunique")).reset_index()
        g["Assiduite"] = g.Heures_reelles / g.Heures_prevues * 100
    elif kind == "reussite":
        g = v.dropna(subset=["Reussis", "Effectif"]).groupby(k).agg(
            Effectif=("Effectif", "sum"), Reussis=("Reussis", "sum"), Filles=("Filles", "sum"), Filles_reussies=("Filles_reussies", "sum"),
            Garcons=("Garcons", "sum"), Garcons_reussis=("Garcons_reussis", "sum")).reset_index()
        g["Reussite"] = g.Reussis / g.Effectif * 100
        g["Part_filles"] = g.Filles / g.Effectif * 100
        g["Reussite_filles"] = g.Filles_reussies / g.Filles.replace(0, np.nan) * 100
        g["Reussite_garcons"] = g.Garcons_reussis / g.Garcons.replace(0, np.nan) * 100
    else:
        g = v.groupby(k).agg(Manque_classes=("n", "sum")).reset_index()
    return g


LEVELS = {"Établissement × Matière": ["eid", "Sous_systeme", "Matiere"],
          "Établissement × Niveau × Matière": ["eid", "Sous_systeme", "Niveau", "Matiere"],
          "Établissement": ["eid", "Sous_systeme"]}
FLAGS = {"C": "Couverture", "A": "Assiduite", "R": "Reussite", "M": "Manque_classes"}


def join(cl, level, lack0=True):
    keys, out = LEVELS[level], None
    for n in ["couverture", "reussite", "manque", "assiduite"]:
        if n not in cl: continue
        p = aggregate(n, cl[n], keys)
        out = p if out is None else out.merge(p, on=[x for x in keys if x in out.columns and x in p.columns], how="outer")
    base = ["eid", "Sous_systeme"]
    att = pd.concat([d[base + ["Etablissement", "Departement", "Arrondissement", "Type_education"]] for d in cl.values()]).drop_duplicates(base)
    out = out.merge(att, on=base, how="left")
    if "Matiere" in out:
        mat = pd.concat([d[["Matiere", "Inspection"]] for d in cl.values()]).drop_duplicates("Matiere")
        out = out.merge(mat, on="Matiere", how="left")
    if "Niveau" in out: out["Cycle"] = out.Niveau.map(cycle)
    out["Sources"] = out.apply(lambda r: "".join(k for k, c in FLAGS.items() if c in out.columns and pd.notna(r[c])), axis=1)
    if "manque" in cl:
        me = cl["manque"].groupby(base).size().rename("Manque_etab").reset_index()
        out = out.merge(me, on=base, how="left")
        if lack0: out[["Manque_classes", "Manque_etab"]] = out[["Manque_classes", "Manque_etab"]].fillna(0)
        out["Manque_ens"] = out.Manque_classes.gt(0).map({True: "Oui", False: "Non"}).where(out.Manque_classes.notna())
    return out


# ---------- Statistiques ----------
def desc(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty: return {}
    m = s.mean()
    return {"N": len(s), "Moyenne": m, "Médiane": s.median(), "Écart-type": s.std(), "CV (%)": s.std() / m * 100 if m else np.nan,
            "Min": s.min(), "Q1": s.quantile(.25), "Q3": s.quantile(.75), "Max": s.max()}


def group_table(kind, d, dim):
    v = valid(kind, d)
    if kind == "manque":
        g = v.groupby(dim).agg(Classes_sans_enseignant=("n", "sum"), Etablissements=("eid", "nunique"), Matieres=("Matiere", "nunique")).reset_index()
        g["Part (%)"] = g.Classes_sans_enseignant / g.Classes_sans_enseignant.sum() * 100
        return g.sort_values("Classes_sans_enseignant", ascending=False)
    num, den = CFG[kind]
    g = v.groupby(dim).agg(N=("taux", "size"), Moyenne=("taux", "mean"), Médiane=("taux", "median"), Num=(num, "sum"), Den=(den, "sum")).reset_index()
    g["Taux pondéré (%)"] = g.Num / g.Den * 100
    return g.drop(columns=["Num", "Den"]).sort_values("Taux pondéré (%)", ascending=False)


def group_test(df, y, g, minn=2):
    gr = [(k, x[y].dropna()) for k, x in df.groupby(g)]
    gr = [(k, s) for k, s in gr if len(s) >= minn]
    if len(gr) < 2: return None
    n, k = sum(len(s) for _, s in gr), len(gr)
    if k == 2:
        a, b = gr[0][1], gr[1][1]
        u, p = stats.mannwhitneyu(a, b)
        return dict(test="Mann-Whitney U", stat=u, p=p, effet=abs(1 - 2 * u / (len(a) * len(b))), effet_nom="r (rang-bisérial)", n=n, k=k)
    h, p = stats.kruskal(*[s for _, s in gr])
    return dict(test="Kruskal-Wallis H", stat=h, p=p, effet=max((h - k + 1) / (n - k), 0), effet_nom="ε²", n=n, k=k)


def posthoc(df, y, g, top=8):
    cnt = df.groupby(g)[y].count().sort_values(ascending=False)
    gr = {k: df.loc[df[g] == k, y].dropna() for k in cnt[cnt >= 2].index[:top]}
    rows = [(a, b, stats.mannwhitneyu(gr[a], gr[b])[1]) for a, b in itertools.combinations(gr, 2)]
    if not rows: return pd.DataFrame()
    adj = multipletests([r[2] for r in rows], method="holm")[1]
    return pd.DataFrame({"Groupe A": [r[0] for r in rows], "Groupe B": [r[1] for r in rows], "p brute": [r[2] for r in rows],
                         "p ajustée (Holm)": adj}).sort_values("p ajustée (Holm)")


def assoc(df, a, b):
    ct = pd.crosstab(df[a], df[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2: return None, None
    chi, p, dof, exp = stats.chi2_contingency(ct)
    n = ct.values.sum()
    r = dict(chi2=chi, p=p, dof=dof, n=n, V=np.sqrt(chi / (n * (min(ct.shape) - 1))), exp_lt5=(exp < 5).mean())
    if ct.shape == (2, 2): r["fisher_p"] = stats.fisher_exact(ct)[1]
    return ct, r


def corr_tables(df, cols, method="Spearman"):
    duplicates = [c for c in dict.fromkeys(cols) if (df.columns == c).sum() > 1]
    if duplicates:
        raise ValueError(
            "Colonnes dupliquées dans le tableau joint : "
            + ", ".join(map(str, duplicates))
            + ". Renommez ou supprimez les colonnes en double avant de calculer les corrélations."
        )
    fn = stats.spearmanr if method == "Spearman" else stats.pearsonr
    r = pd.DataFrame(np.nan, index=cols, columns=cols); p = r.copy(); n = r.copy()
    for a in cols:
        for b in cols:
            x = df[[a, b]].dropna(); n.loc[a, b] = len(x)
            xa, xb = x.iloc[:, 0], x.iloc[:, 1]
            if len(x) > 2 and xa.nunique() > 1 and xb.nunique() > 1:
                r.loc[a, b], p.loc[a, b] = fn(xa, xb)
    return r, p, n


def vif(X):
    if X.shape[1] < 2: return pd.Series(dtype=float)
    M = sm.add_constant(X, has_constant="add").values
    return pd.Series([variance_inflation_factor(M, i) for i in range(1, M.shape[1])], index=X.columns)


# ---------- Modèle ----------
MODELS = ["MCO (erreurs robustes HC3)", "MCO pondérés par l'effectif", "Logit binomial (réussis / effectif)"]


def design(df, y, xn, xc):
    d = df[list(dict.fromkeys([y] + xn + xc + ["Effectif", "Reussis"]))].dropna(subset=[y] + xn + xc).copy()
    X = pd.get_dummies(d[xn + xc], columns=xc, drop_first=True, dtype=float)
    return d, X.loc[:, X.nunique() > 1]


def fit(df, y, xn, xc, kind):
    d, X = design(df, y, xn, xc)
    Xc = sm.add_constant(X, has_constant="add")
    glm = kind == MODELS[2]
    if kind == MODELS[0]: res = sm.OLS(d[y], Xc).fit(cov_type="HC3")
    elif kind == MODELS[1]: res = sm.WLS(d[y], Xc, weights=d["Effectif"]).fit(cov_type="HC3")
    else: res = sm.GLM(np.column_stack([d["Reussis"], d["Effectif"] - d["Reussis"]]), Xc, family=sm.families.Binomial()).fit(cov_type="HC1")
    ci = res.conf_int()
    tab = pd.DataFrame({"Coef.": res.params, "Écart-type": res.bse, "z / t": res.tvalues, "p-value": res.pvalues,
                        "IC95 bas": ci[0], "IC95 haut": ci[1]})
    if glm: tab["Odds ratio"] = np.exp(tab["Coef."])
    dg = {"N": int(res.nobs), "Variables (k)": X.shape[1]}
    if glm:
        dg["Pseudo-R² (déviance)"] = 1 - res.deviance / res.null_deviance; dg["AIC"] = res.aic
        resid, fitted = res.resid_pearson, res.fittedvalues * 100
    else:
        dg.update({"R²": res.rsquared, "R² ajusté": res.rsquared_adj, "AIC": res.aic, "BIC": res.bic})
        resid, fitted = res.resid, res.fittedvalues
        if kind == MODELS[0]:
            try: dg["Breusch-Pagan p"] = het_breuschpagan(res.resid, res.model.exog)[1]
            except Exception: pass
    if len(resid) >= 3: dg["Shapiro-Wilk p (résidus)"] = stats.shapiro(resid)[1]
    return dict(tab=tab, diag=dg, resid=np.asarray(resid), fitted=np.asarray(fitted), vif=vif(X), glm=glm)


# ---------- Analyse multi-niveau ----------
GEO = ["Region", "Departement", "Arrondissement", "Etablissement"]
RATE = {"couverture": "Couverture", "assiduite": "Assiduite", "reussite": "Reussite"}


def cells(cl):
    """Taux par cellule (établissement × sous-système × matière), avec la géographie complète."""
    return {k: aggregate(k, cl[k], GEO + ["Sous_systeme", "Matiere"]) for k in RATE if k in cl}


def level_means(cells_, by):
    """Taux par unité géographique : moyenne des taux des cellules établissement × matière.
    Équivaut à pondérer chaque établissement par son nombre de matières."""
    out = None
    for k, c in cells_.items():
        col = RATE[k]; c = c[c[col].notna()]
        g = c.groupby(by).agg(**{col: (col, "mean"), "Cellules_" + col: (col, "size")}).reset_index()
        out = g if out is None else out.merge(g, on=by, how="outer")
    return out


def below(c, seuil, by):
    c = c.assign(sous=c.Reussite < seuil)
    g = c.groupby(by).agg(Matieres_total=("Reussite", "size"), Matieres_sous_seuil=("sous", "sum"), Reussite_moyenne=("Reussite", "mean")).reset_index()
    g["Part sous le seuil (%)"] = g.Matieres_sous_seuil / g.Matieres_total * 100
    return g.sort_values("Part sous le seuil (%)", ascending=False)
