import io
import numpy as np, pandas as pd, streamlit as st
import etl, viz

st.set_page_config(page_title="Observatoire des indicateurs éducatifs", page_icon=":material/school:", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1400px}
h1,h2,h3{color:#1F3A5F;letter-spacing:-.01em}
[data-testid="stMetricValue"]{font-size:1.5rem;color:#1F3A5F}
[data-testid="stSidebar"]{border-right:1px solid #E3E7ED}
footer,#MainMenu{visibility:hidden}
</style>""", unsafe_allow_html=True)

LABEL = {"couverture": "Taux de couverture des programmes", "assiduite": "Taux d'assiduité",
         "manque": "Manque d'enseignants", "reussite": "Taux de réussite (1re séquence)"}
DIMS = ["Departement", "Arrondissement", "Sous_systeme", "Type_education", "Inspection", "Matiere", "Etablissement", "Cycle", "Niveau"]
NUMS = ["Couverture", "Assiduite", "Reussite", "Manque_classes", "Manque_etab", "Effectif", "Part_filles", "Heures_prevues", "Nb_enseignants"]
CATS = ["Departement", "Arrondissement", "Sous_systeme", "Type_education", "Inspection", "Matiere", "Cycle", "Niveau", "Manque_ens"]


def table(df, **kw):
    num = df.select_dtypes("number").columns
    st.dataframe(df.round({c: 3 if "p" in c.lower() else 2 for c in num}), width="stretch", hide_index=True, **kw)


def pfmt(p): return "< 0,001" if p < .001 else f"{p:.3f}"


def verdict(p, a=.05):
    (st.success if p < a else st.info)(f"{'Différence / association significative' if p < a else 'Pas de preuve statistique'} au seuil de {int(a*100)} % (p = {pfmt(p)}).", icon=":material/rule:")


# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### :material/school: Observatoire éducatif")
    st.caption("Suivi des indicateurs d'inspection pédagogique")
    up = st.file_uploader("Importer les 4 fichiers Excel", type=["xlsx"], accept_multiple_files=True,
                          help="Manque d'enseignants, couverture des programmes, assiduité, réussite. Ordre indifférent : les fichiers sont reconnus par leurs colonnes.")
    st.markdown("##### :material/tune: Paramètres")
    level = st.selectbox("Niveau de jointure", list(etl.LEVELS), help="Grain du tableau joint. L'assiduité (sans classe) est répliquée sur les niveaux.")
    cap = st.toggle("Plafonner l'assiduité à 100 %", True)
    seuil = st.number_input("Seuil d'aberration assiduité (%)", 100, 2000, 150, step=10, help="Lignes dont heures réelles / prévues ≥ seuil : exclues.")
    lack0 = st.toggle("Absence dans « manque » = 0", True, help="Le fichier liste uniquement les manques déclarés ; l'absence d'une ligne est lue comme « pas de manque ».")


@st.cache_data(show_spinner="Lecture et harmonisation des fichiers…")
def load(files, cap, seuil):
    cl, log = {}, []
    for name, b in files:
        df = pd.read_excel(io.BytesIO(b)); k = etl.detect(df)
        if k: cl[k] = etl.clean(k, df, cap, seuil)
        log.append((name, k or "non reconnu", len(df)))
    return cl, log


if not up:
    st.title("Observatoire des indicateurs éducatifs")
    st.markdown("Importez les quatre fichiers dans la barre latérale pour lancer les analyses : **harmonisation et jointure**, **statistiques univariées**, **analyses croisées avec tests** et **modélisation économétrique** du taux de réussite.")
    st.info("Les fichiers sont identifiés automatiquement par leurs colonnes.", icon=":material/upload_file:")
    st.stop()

cl, log = load(tuple((f.name, f.getvalue()) for f in up), cap, seuil)
joined = etl.join(cl, level, lack0) if cl else pd.DataFrame()
NUMJ = [c for c in NUMS if c in joined]; CATJ = [c for c in CATS if c in joined]


# ---------------- Pages ----------------
def page_data():
    st.header(":material/database: Données et jointure")
    missing = [LABEL[k] for k in LABEL if k not in cl]
    if missing: st.warning("Fichier(s) non détecté(s) : " + ", ".join(missing), icon=":material/warning:")
    c = st.columns(len(cl))
    for col, (k, d) in zip(c, cl.items()):
        col.metric(LABEL[k], f"{len(d):,} lignes".replace(",", " "), f"{d.eid.nunique()} établissements · {d.Matiere.nunique()} matières", delta_color="off", border=True)
    st.subheader("Contrôle qualité et harmonisation")
    q = []
    for k, d in cl.items():
        q.append({"Fichier": LABEL[k], "Lignes": len(d), "Matières brutes": d.Matiere_brute.nunique(), "Matières harmonisées": d.Matiere.nunique(),
                  "Anomalies": {"assiduite": f"{int(d.aberrant.sum())} lignes ≥ seuil", "couverture": f"{int((d.Prevu == 0).sum())} lignes à Prévu = 0",
                              "reussite": f"{int(d.taux.isna().sum())} ligne(s) incomplète(s)", "manque": "—"}[k]})
    table(pd.DataFrame(q))
    st.caption("Harmonisation : sous-système (Anglophone/Francophone), type d'éducation (Général/Technique/Normal), établissements (casse et ponctuation), matières FR/EN fusionnées, classes ramenées au niveau (Form 4 All → Form 4 ; Terminale A/C/D → Terminale) et au cycle. Les taux sont recalculés à partir des effectifs et heures.")
    st.subheader("Tableau joint")
    st.caption(f"Grain : {level}. Les taux agrégés sont pondérés (somme du réalisé / somme du prévu).")
    t = joined.Sources.value_counts().rename_axis("Sources présentes").reset_index(name="Lignes")
    cA, cB = st.columns([1, 2])
    with cA:
        table(t); st.caption("C = couverture · A = assiduité · R = réussite · M = manque d'enseignants")
    with cB:
        st.metric("Lignes du tableau joint", len(joined), border=True)
        st.metric("Lignes avec réussite + couverture + assiduité", int(joined.Sources.str.contains("C") .mul(joined.Sources.str.contains("A")).mul(joined.Sources.str.contains("R")).sum()), border=True)
    table(joined)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        joined.to_excel(w, "Tableau_joint", index=False)
        for k, d in cl.items(): d.to_excel(w, k[:30], index=False)
    b1, b2 = st.columns(2)
    b1.download_button("Télécharger le tableau joint (CSV)", joined.to_csv(index=False).encode("utf-8-sig"), "tableau_joint.csv", "text/csv", icon=":material/download:")
    b2.download_button("Télécharger le classeur (joint + fichiers nettoyés)", buf.getvalue(), "indicateurs_harmonises.xlsx", icon=":material/download:")
    with st.expander(":material/translate: Table de correspondance des matières"):
        m = pd.concat([d[["Matiere_brute", "Matiere"]] for d in cl.values()]).drop_duplicates()
        table(m[m.groupby("Matiere").Matiere_brute.transform("nunique") > 1].sort_values("Matiere"))


def uni_tab():
    c1, c2, c3 = st.columns([2, 2, 1])
    kind = c1.selectbox("Indicateur", [k for k in LABEL if k in cl], format_func=LABEL.get)
    d = cl[kind]; v = etl.valid(kind, d)
    dim = c2.selectbox("Découpage", [x for x in DIMS if x in d.columns])
    top = c3.number_input("Top N", 5, 40, 12)
    g = etl.group_table(kind, d, dim)
    if kind != "manque":
        num, den = etl.CFG[kind]; s = v["taux"]; ds = etl.desc(s)
        k = st.columns(5)
        k[0].metric("Observations", ds["N"], border=True); k[1].metric("Taux global pondéré", f"{v[num].sum() / v[den].sum() * 100:.1f} %", border=True)
        k[2].metric("Médiane", f"{ds['Médiane']:.1f} %", border=True); k[3].metric("Écart-type", f"{ds['Écart-type']:.1f}", border=True); k[4].metric("CV", f"{ds['CV (%)']:.0f} %", border=True)
        a, b = st.columns(2)
        with a: viz.hist(f"Distribution – {LABEL[kind]}", s, f"h_{kind}", xlabel="%")
        with b: viz.box(f"{LABEL[kind]} par {dim}", v, "taux", dim, f"b_{kind}_{dim}", top, "%")
        gg = g.head(top)
        viz.bar(f"{LABEL[kind]} pondéré par {dim}", [f"{x} (n={n})" for x, n in zip(gg[dim], gg.N)], gg["Taux pondéré (%)"], f"g_{kind}_{dim}", "%", horizontal=True)
        r = etl.group_test(v, "taux", dim)
        st.subheader("Test de différence entre groupes")
        if r:
            m = st.columns(4); m[0].metric("Test", r["test"], border=True); m[1].metric("Statistique", f"{r['stat']:.2f}", border=True)
            m[2].metric("p-value", pfmt(r["p"]), border=True); m[3].metric(f"Taille d'effet ({r['effet_nom']})", f"{r['effet']:.2f}", border=True)
            verdict(r["p"])
        else: st.info("Effectifs insuffisants pour un test.")
        st.subheader("Tableau détaillé"); table(g)
    else:
        k = st.columns(3)
        k[0].metric("Classes sans enseignant", len(d), border=True); k[1].metric("Établissements concernés", d.eid.nunique(), border=True); k[2].metric("Matières concernées", d.Matiere.nunique(), border=True)
        gg = g.head(top)
        viz.bar(f"Classes sans enseignant par {dim}", gg[dim], gg.Classes_sans_enseignant, f"m_{dim}", "Classes", horizontal=True)
        st.subheader("Tableau détaillé"); table(g)
        st.caption("Le fichier ne contient pas de dénominateur (effectifs d'enseignants requis) : l'analyse porte sur les volumes et leur répartition.")


def xl(df, name, key):
    b = io.BytesIO()
    with pd.ExcelWriter(b, engine="xlsxwriter") as w: df.to_excel(w, sheet_name=name[:31], index=False)
    st.download_button("Exporter en Excel", b.getvalue(), f"{name}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", icon=":material/download:", key=key)


def table_xl(df, name, key):
    if df.empty: st.info("Aucune ligne pour ces filtres.", icon=":material/filter_alt_off:"); return
    table(df); xl(df, name, key)


def geo_apply(df, reg=None, dep=None, arr=None, etab=None):
    if reg and reg != "Toutes": df = df[df.Region == reg]
    if dep and dep != "Tous": df = df[df.Departement == dep]
    if arr and arr != "Tous": df = df[df.Arrondissement == arr]
    if etab: df = df[df.Etablissement.isin(etab)]
    return df


def multi_tab():
    loc = pd.concat([d[etl.GEO] for d in cl.values()]).drop_duplicates()
    c = st.columns(4)
    reg = c[0].selectbox("Région", ["Toutes"] + sorted(loc.Region.unique()), key="mt_reg")
    L = loc if reg == "Toutes" else loc[loc.Region == reg]
    dep = c[1].selectbox("Département", ["Tous"] + sorted(L.Departement.unique()), key=f"mt_dep_{reg}")
    L = L if dep == "Tous" else L[L.Departement == dep]
    arr = c[2].selectbox("Arrondissement", ["Tous"] + sorted(L.Arrondissement.unique()), key=f"mt_arr_{reg}_{dep}")
    L = L if arr == "Tous" else L[L.Arrondissement == arr]
    etab = c[3].multiselect("Établissements", sorted(L.Etablissement.unique()), placeholder="Tous les établissements", key=f"mt_et_{reg}_{dep}_{arr}")
    F = dict(reg=reg, dep=dep, arr=arr, etab=etab)
    cl_cells = etl.cells(cl); NM = {"couverture": "Couverture des programmes", "assiduite": "Assiduité", "reussite": "Réussite"}

    st.subheader("Indicateurs agrégés")
    st.caption("Chaque cellule établissement × matière compte pour une unité : l'agrégat est la moyenne de ces taux, soit une pondération de chaque établissement par son nombre de matières.")
    for by, filt, title, key, note in [
        (["Departement"], dict(reg=reg), "Indicateurs par département", "mt_c1", "Non affecté par le filtre département (ni arrondissement / établissement)."),
        (["Departement", "Arrondissement"], dict(reg=reg, dep=dep), "Indicateurs par arrondissement", "mt_c2", "Affecté par les filtres région et département.")]:
        m = etl.level_means({k: geo_apply(x, **filt) for k, x in cl_cells.items()}, by)
        if m is None or m.empty: st.info("Aucune donnée pour ces filtres."); continue
        viz.bars_multi(title, m[by[-1]], {NM[k]: m[v] for k, v in etl.RATE.items() if v in m}, key, sub=note)
        table_xl(m, title.replace(" ", "_").lower(), key + "_x")

    st.subheader("Tableaux détaillés")
    specs = [("reussite", ["Matiere"], ["Effectif", "Reussis", "Reussite", "Part_filles"], "Réussite par matière", ":material/school:"),
             ("couverture", ["Matiere", "Ens"], ["Prevu", "Realise", "Couverture"], "Couverture par enseignant", ":material/menu_book:"),
             ("assiduite", ["Matiere", "Ens"], ["Heures_prevues", "Heures_reelles", "Assiduite"], "Assiduité par enseignant", ":material/schedule:")]
    specs = [s_ for s_ in specs if s_[0] in cl]
    for tab, (kind, extra, cols, label, icon) in zip(st.tabs([f"{i} {l}" for _, _, _, l, i in specs]), specs):
        with tab:
            d = geo_apply(cl[kind], **F).assign(Ens=lambda x: x["Enseignant"] if "Enseignant" in x else "")
            g = etl.aggregate(kind, d, etl.GEO + extra) if len(d) else pd.DataFrame()
            if len(g):
                g = g.rename(columns={"Ens": "Enseignant"}); ex = [e if e != "Ens" else "Enseignant" for e in extra]
                g = g[etl.GEO[1:] + ex + cols].sort_values(etl.RATE[kind] if etl.RATE[kind] in cols else cols[0])
            table_xl(g, label.replace(" ", "_").lower(), f"mt_t_{kind}")

    st.subheader("Matières sous un seuil de réussite")
    if "reussite" not in cl_cells: st.info("Fichier de réussite absent."); return
    seuil = st.slider("Seuil de taux de réussite (%)", 0, 100, 50, key="mt_seuil")
    rc = geo_apply(cl_cells["reussite"], **F); sous = rc[rc.Reussite < seuil]
    k = st.columns(3); k[0].metric("Cellules établissement × matière", len(rc), border=True)
    k[1].metric("Sous le seuil", len(sous), border=True); k[2].metric("Part", f"{len(sous) / len(rc) * 100:.0f} %" if len(rc) else "—", border=True)
    views = [("Détail matière", None), ("Par département", ["Departement"]), ("Par arrondissement", ["Departement", "Arrondissement"]), ("Par établissement", ["Departement", "Arrondissement", "Etablissement"])]
    for tab, (lab, by) in zip(st.tabs([v[0] for v in views]), views):
        with tab:
            out = sous.sort_values("Reussite")[["Departement", "Arrondissement", "Etablissement", "Sous_systeme", "Matiere", "Effectif", "Reussis", "Reussite"]] if by is None else (etl.below(rc, seuil, by) if len(rc) else pd.DataFrame())
            if by is not None and len(out): out = out[out.Matieres_sous_seuil > 0]
            table_xl(out, f"sous_seuil_{lab.lower().replace(' ', '_')}", f"mt_s_{lab}")


def page_uni():
    st.header(":material/bar_chart: Analyse univariée")
    t1, t2 = st.tabs([":material/analytics: Indicateurs", ":material/account_tree: Analyse multi-niveau"])
    with t1: uni_tab()
    with t2: multi_tab()


def page_cross():
    st.header(":material/compare_arrows: Analyses croisées")
    st.caption(f"Base : tableau joint, grain « {level} ».")
    J = joined.copy()
    for c in ["Couverture", "Assiduite", "Reussite"]:
        if c in J and J[c].notna().sum() > 6:
            try: J[c + "_tercile"] = pd.qcut(J[c], 3, labels=["Bas", "Moyen", "Haut"])
            except ValueError: pass
    cats = CATJ + [c for c in J if c.endswith("_tercile")]
    t1, t2, t3, t4 = st.tabs([":material/grid_on: Corrélations", ":material/stacked_bar_chart: Comparaison de groupes", ":material/table_chart: Associations qualitatives", ":material/person_off: Effet du manque d'enseignants"])
    with t1:
        cols = st.multiselect("Variables", NUMJ, default=[c for c in ["Couverture", "Assiduite", "Reussite", "Manque_classes", "Effectif"] if c in NUMJ])
        meth = st.radio("Méthode", ["Spearman", "Pearson"], horizontal=True)
        if len(cols) >= 2:
            r, p, n = etl.corr_tables(J, cols, meth); viz.heat(f"Matrice de corrélation ({meth})", r, "heat")
            pairs = [{"Variable 1": a, "Variable 2": b, "Coefficient": r.loc[a, b], "p-value": p.loc[a, b], "N": int(n.loc[a, b])} for i, a in enumerate(cols) for b in cols[i + 1:]]
            table(pd.DataFrame(pairs))
            x, y = st.columns(2); xv = x.selectbox("Axe X", cols); yv = y.selectbox("Axe Y", cols, index=1)
            viz.scatter(f"{yv} selon {xv}", J, xv, yv, "sc")
    with t2:
        a, b = st.columns(2); y = a.selectbox("Variable quantitative", NUMJ, key="gy"); g = b.selectbox("Regroupement", cats, key="gg")
        D = J[[y, g]].dropna(); r = etl.group_test(D, y, g)
        if r is None: st.info("Effectifs insuffisants.")
        else:
            m = st.columns(4); m[0].metric("Test", r["test"], border=True); m[1].metric("p-value", pfmt(r["p"]), border=True)
            m[2].metric(r["effet_nom"], f"{r['effet']:.2f}", border=True); m[3].metric("N", r["n"], border=True)
            verdict(r["p"]); viz.box(f"{y} selon {g}", D, y, g, "bx", 12)
            table(D.groupby(g)[y].agg(N="count", Moyenne="mean", Médiane="median", Écart_type="std").reset_index())
            if r["k"] > 2 and r["p"] < .05:
                st.markdown("**Comparaisons deux à deux (Mann-Whitney, correction de Holm)**"); table(etl.posthoc(D, y, g))
    with t3:
        a, b = st.columns(2); v1 = a.selectbox("Variable 1", cats, key="c1"); v2 = b.selectbox("Variable 2", cats, index=min(1, len(cats) - 1), key="c2")
        D = J[[v1, v2]].dropna()
        ct, r = etl.assoc(D, v1, v2) if v1 != v2 else (None, None)
        if r is None: st.info("Choisissez deux variables distinctes avec au moins deux modalités.")
        else:
            m = st.columns(4); m[0].metric("χ²", f"{r['chi2']:.2f}", border=True); m[1].metric("p-value", pfmt(r["p"]), border=True)
            m[2].metric("V de Cramér", f"{r['V']:.2f}", border=True); m[3].metric("Cellules attendues < 5", f"{r['exp_lt5']:.0%}", border=True)
            verdict(r["p"])
            if r["exp_lt5"] > .2: st.warning("Plus de 20 % des effectifs théoriques sont < 5 : le χ² est peu fiable, regrouper les modalités.", icon=":material/warning:")
            if "fisher_p" in r: st.caption(f"Test exact de Fisher : p = {pfmt(r['fisher_p'])}")
            table(ct.reset_index())
            prof = (ct.div(ct.sum(axis=1), axis=0) * 100).round(1)
            viz.bar(f"Profil en lignes : {v2} selon {v1} (%)", prof.index, prof.iloc[:, 0], "as", f"% {prof.columns[0]}")
    with t4:
        if "Manque_ens" not in J: st.info("Fichier « manque d'enseignants » absent.")
        else:
            rows = []
            for c in ["Couverture", "Assiduite", "Reussite"]:
                if c in J:
                    D = J[[c, "Manque_ens"]].dropna(); r = etl.group_test(D, c, "Manque_ens")
                    med = D.groupby("Manque_ens")[c].median()
                    rows.append({"Indicateur": c, "Médiane (manque = Non)": med.get("Non"), "Médiane (manque = Oui)": med.get("Oui"),
                                 "N Non": int((D.Manque_ens == "Non").sum()), "N Oui": int((D.Manque_ens == "Oui").sum()),
                                 "p-value (Mann-Whitney)": r["p"] if r else np.nan, "r": r["effet"] if r else np.nan})
            R = pd.DataFrame(rows); table(R)
            if len(R): viz.bar("Médianes selon la présence d'un manque d'enseignants", R.Indicateur, R["Médiane (manque = Oui)"], "lk", "%", sub="Barres : cas avec manque (Oui)")
            st.caption("Un manque déclaré a une valeur explicative seulement si l'établissement × matière figure aussi dans les autres fichiers (voir « Sources présentes »).")


def page_model():
    st.header(":material/functions: Modélisation du taux de réussite")
    J = joined
    if "Reussite" not in J or J.Reussite.notna().sum() < 5:
        st.warning("Le taux de réussite est absent ou trop peu renseigné au grain choisi.", icon=":material/warning:"); return
    st.markdown("##### 1. Choix des variables explicatives")
    st.caption("Variables principales présélectionnées : couverture, assiduité, manque d'enseignants. Ajoutez des variables de contrôle après examen du tableau de dépistage.")
    xn_all = [c for c in NUMJ if c not in ("Reussite", "Manque_etab") or c == "Manque_etab"]; xn_all = [c for c in xn_all if c != "Reussite"]
    xc_all = [c for c in CATJ if c not in ("Matiere", "Arrondissement", "Niveau")]
    a, b = st.columns(2)
    xn = a.multiselect("Variables quantitatives", xn_all, default=[c for c in ["Couverture", "Assiduite", "Manque_classes"] if c in xn_all])
    xc = b.multiselect("Variables qualitatives", xc_all, default=[])
    scr = []
    for c in xn_all:
        D = J[["Reussite", c]].dropna()
        if len(D) > 3 and D[c].nunique() > 1:
            r, p = etl.stats.spearmanr(D.Reussite, D[c]); scr.append({"Variable": c, "Type": "Quanti", "Statistique": r, "p-value": p, "N": len(D)})
    for c in xc_all:
        D = J[["Reussite", c]].dropna(); r = etl.group_test(D, "Reussite", c)
        if r: scr.append({"Variable": c, "Type": "Quali", "Statistique": r["effet"], "p-value": r["p"], "N": r["n"]})
    with st.expander(":material/manage_search: Tableau de dépistage (corrélation de Spearman / test de groupes avec la réussite)", expanded=True):
        table(pd.DataFrame(scr)); st.caption("Statistique : ρ de Spearman (quanti) ou taille d'effet (quali).")
    if not (xn or xc): st.info("Sélectionnez au moins une variable."); return
    D, X = etl.design(J, "Reussite", xn, xc)
    k = st.columns(3); k[0].metric("Observations complètes", len(D), border=True); k[1].metric("Paramètres estimés", X.shape[1], border=True)
    k[2].metric("Obs. par paramètre", f"{len(D) / max(X.shape[1], 1):.1f}", border=True)
    if len(D) < 10 * max(X.shape[1], 1): st.warning("Moins de 10 observations par paramètre : estimations instables, interpréter avec prudence.", icon=":material/warning:")
    if X.shape[1] >= 2 and len(D) > X.shape[1] + 2:
        vf = etl.vif(X).rename("VIF").reset_index().rename(columns={"index": "Variable"}); table(vf)
        if (vf.VIF > 5).any(): st.warning("VIF > 5 : colinéarité à examiner (retirer ou regrouper une variable).", icon=":material/warning:")
    st.markdown("##### 2. Spécification et estimation")
    kind = st.radio("Modèle", etl.MODELS, horizontal=True, help="Le logit binomial modélise directement réussis / effectif ; les MCO pondérés donnent plus de poids aux grands effectifs.")
    if st.button("Estimer le modèle", type="primary", icon=":material/play_arrow:"):
        try: st.session_state["res"] = (kind, etl.fit(J, "Reussite", xn, xc, kind))
        except Exception as e: st.error(f"Estimation impossible : {e}", icon=":material/error:")
    if "res" in st.session_state:
        kd, r = st.session_state["res"]; st.markdown(f"##### 3. Résultats — {kd}")
        m = st.columns(len(r["diag"]))
        for col, (n_, v_) in zip(m, r["diag"].items()): col.metric(n_, f"{v_:.3f}" if isinstance(v_, float) else v_, border=True)
        tab = r["tab"].reset_index().rename(columns={"index": "Variable"}); table(tab)
        sig = [i for i, p in r["tab"]["p-value"].items() if p < .05 and i != "const"]
        st.markdown(("Variables significatives à 5 % : **" + ", ".join(sig) + "**.") if sig else "Aucune variable significative à 5 %.")
        st.caption("Lecture : MCO = variation du taux de réussite en points de % ; logit = log-odds (voir odds ratio).")
        a, b = st.columns(2)
        with a: viz.coefs("Coefficients estimés", r["tab"], "co")
        with b: viz.resid_plot("Résidus selon les valeurs ajustées", r["fitted"], r["resid"], "rs")
        st.download_button("Télécharger les coefficients (CSV)", tab.to_csv(index=False).encode("utf-8-sig"), "coefficients_modele.csv", icon=":material/download:")


pg = st.navigation([
    st.Page(page_data, title="Données et jointure", icon=":material/database:", url_path="donnees"),
    st.Page(page_uni, title="Analyse univariée", icon=":material/bar_chart:", url_path="univariee"),
    st.Page(page_cross, title="Analyses croisées", icon=":material/compare_arrows:", url_path="croisees"),
    st.Page(page_model, title="Modélisation", icon=":material/functions:", url_path="modele"),
])
pg.run()
