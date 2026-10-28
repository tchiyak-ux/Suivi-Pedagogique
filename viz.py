"""Graphiques ECharts (téléchargeables : PNG haute résolution + vue des données)."""
import re
import numpy as np, pandas as pd
from streamlit_echarts import st_echarts

PAL = ["#1F3A5F", "#2A9D8F", "#E9C46A", "#E76F51", "#6C757D", "#457B9D", "#8AB17D", "#B5838D"]


def _f(v):
    try: return None if pd.isna(v) else round(float(v), 2)
    except Exception: return None


def _base(title, sub=""):
    return {"color": PAL,
            "title": {"text": title, "subtext": sub, "left": "left", "textStyle": {"fontSize": 15, "fontWeight": 600}},
            "toolbox": {"right": 10, "feature": {
                "saveAsImage": {"title": "Télécharger (PNG)", "name": re.sub(r"\W+", "_", title), "pixelRatio": 3, "backgroundColor": "#fff"},
                "dataView": {"title": "Données", "readOnly": True, "lang": ["Données", "Fermer", "Actualiser"]}}},
            "tooltip": {"trigger": "axis"},
            "grid": {"left": 40, "right": 30, "top": 80, "bottom": 60, "containLabel": True}}


def draw(o, key, h=420):
    st_echarts(o, height=f"{h}px", key=key)


def bar(title, cats, vals, key, ylabel="", horizontal=False, sub=""):
    cats = [str(c) for c in cats]; o = _base(title, sub)
    cat = {"type": "category", "data": cats, "axisLabel": {"interval": 0, "rotate": 0 if horizontal else 35}}
    val = {"type": "value", "name": ylabel}
    o["xAxis"], o["yAxis"] = (val, {**cat, "inverse": True}) if horizontal else (cat, val)
    o["series"] = [{"type": "bar", "data": [_f(v) for v in vals], "label": {"show": True, "position": "right" if horizontal else "top"}}]
    draw(o, key, max(320, 26 * len(cats) + 120) if horizontal else 420)


def hist(title, s, key, bins=15, xlabel=""):
    c, e = np.histogram(pd.Series(s).dropna(), bins=bins)
    o = _base(title)
    o["xAxis"] = {"type": "category", "data": [f"{a:.0f}–{b:.0f}" for a, b in zip(e[:-1], e[1:])], "name": xlabel, "nameLocation": "middle", "nameGap": 40}
    o["yAxis"] = {"type": "value", "name": "Effectif"}
    o["series"] = [{"type": "bar", "data": [int(x) for x in c], "barCategoryGap": "8%"}]
    draw(o, key)


def box(title, df, y, g, key, top=12, ylabel=""):
    cnt = df.groupby(g)[y].count().sort_values(ascending=False).head(top)
    data, out, labs = [], [], []
    for i, c in enumerate(cnt.index):
        s = df.loc[df[g] == c, y].dropna()
        q1, m, q3 = s.quantile([.25, .5, .75]); iqr = q3 - q1
        lo, hi = s[s >= q1 - 1.5 * iqr].min(), s[s <= q3 + 1.5 * iqr].max()
        data.append([_f(lo), _f(q1), _f(m), _f(q3), _f(hi)]); labs.append(f"{c} (n={len(s)})")
        out += [[i, _f(v)] for v in s[(s < lo) | (s > hi)]]
    o = _base(title); o["tooltip"] = {"trigger": "item"}
    o["xAxis"] = {"type": "category", "data": labs, "axisLabel": {"interval": 0, "rotate": 30}}
    o["yAxis"] = {"type": "value", "name": ylabel}
    o["series"] = [{"type": "boxplot", "data": data}, {"type": "scatter", "data": out, "symbolSize": 5}]
    draw(o, key)


def scatter(title, df, x, y, key):
    d = df[[x, y]].dropna(); o = _base(title); o["tooltip"] = {"trigger": "item"}
    o["xAxis"] = {"type": "value", "name": x, "scale": True}; o["yAxis"] = {"type": "value", "name": y, "scale": True}
    o["series"] = [{"type": "scatter", "data": [[_f(a), _f(b)] for a, b in zip(d[x], d[y])], "symbolSize": 8}]
    if len(d) > 2 and d[x].nunique() > 1:
        b, a = np.polyfit(d[x], d[y], 1); x0, x1 = d[x].min(), d[x].max()
        o["series"].append({"type": "line", "data": [[_f(x0), _f(a + b * x0)], [_f(x1), _f(a + b * x1)]], "showSymbol": False, "lineStyle": {"color": "#E76F51"}})
    draw(o, key)


def heat(title, corr, key):
    labs = [str(c) for c in corr.columns]
    data = [[j, i, _f(corr.iloc[i, j])] for i in range(len(labs)) for j in range(len(labs)) if not pd.isna(corr.iloc[i, j])]
    o = _base(title); o["tooltip"] = {"position": "top"}; o["grid"]["bottom"] = 90
    o["xAxis"] = {"type": "category", "data": labs, "axisLabel": {"rotate": 30, "interval": 0}}
    o["yAxis"] = {"type": "category", "data": labs, "inverse": True}
    o["visualMap"] = {"min": -1, "max": 1, "calculable": True, "orient": "horizontal", "left": "center", "bottom": 0,
                      "inRange": {"color": ["#E76F51", "#FFFFFF", "#1F3A5F"]}}
    o["series"] = [{"type": "heatmap", "data": data, "label": {"show": True}}]
    draw(o, key, max(380, 60 * len(labs) + 140))


def coefs(title, tab, key):
    t = tab.drop(index="const", errors="ignore")
    o = _base(title, "Barres : coefficients estimés"); o["tooltip"] = {"trigger": "axis"}
    o["xAxis"] = {"type": "value"}; o["yAxis"] = {"type": "category", "data": [str(i) for i in t.index], "inverse": True}
    o["series"] = [{"type": "bar", "data": [{"value": _f(v), "itemStyle": {"color": "#2A9D8F" if v >= 0 else "#E76F51"}} for v in t["Coef."]],
                    "label": {"show": True, "position": "right"}}]
    draw(o, key, max(320, 34 * len(t) + 120))


def resid_plot(title, fitted, resid, key):
    o = _base(title); o["tooltip"] = {"trigger": "item"}
    o["xAxis"] = {"type": "value", "name": "Valeurs ajustées", "scale": True}; o["yAxis"] = {"type": "value", "name": "Résidus", "scale": True}
    o["series"] = [{"type": "scatter", "data": [[_f(a), _f(b)] for a, b in zip(fitted, resid)], "symbolSize": 8,
                    "markLine": {"data": [{"yAxis": 0}], "symbol": "none"}}]
    draw(o, key)


def bars_multi(title, cats, series, key, ylabel="%", sub=""):
    """Barres groupées : plusieurs indicateurs sur la même figure."""
    o = _base(title, sub); o["legend"] = {"bottom": 0}; o["grid"]["bottom"] = 90
    o["xAxis"] = {"type": "category", "data": [str(c) for c in cats], "axisLabel": {"interval": 0, "rotate": 30}}
    o["yAxis"] = {"type": "value", "name": ylabel}
    o["series"] = [{"name": n, "type": "bar", "data": [_f(v) for v in vals], "label": {"show": True, "position": "top", "fontSize": 10}} for n, vals in series.items()]
    draw(o, key, 460)
