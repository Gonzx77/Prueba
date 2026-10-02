"""Probabilidad de lluvia por municipio para enero de 2025.

Dos problemas que este modulo resuelve y el anterior no:

1. La lluvia es fuertemente estacional: la mediana va de 45.7 mm en enero a
   248.8 mm en octubre. Un umbral absoluto (150 mm) es casi imposible de
   alcanzar en enero (4.1%) y trivial en octubre, asi que un mismo umbral
   mezcla dos problemas distintos. Se usa un umbral RELATIVO al calendario:
   un mes es "lluvioso" cuando su lluvia supera el percentil 75 de ese mismo
   mes, calculado solo con años previos.

2. Los umbrales por mes se calculan exclusivamente con años anteriores al
   que se predice. En el backtest cada corte recalcula sus umbrales con lo
   que hay hasta el año previo, de modo que un enero de 2019 nunca se evalua
   contra un umbral que ya miro ese mismo enero.

La frontera de informacion es la misma que en el resto del proyecto: para el
mes t solo entra informacion hasta t-1.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from modelo import FEATURES, RegresionLogistica, _anexar_filas_futuras, _ensamble, construir_features

BASE_DIR = Path(__file__).resolve().parent
RUTA_PANEL = BASE_DIR / "Datos" / "panel_entrenamiento_2015_2024.csv"

PERCENTIL = 75
OBJETIVO = "mes_lluvioso"

ANIO_ENTRENAMIENTO = range(2016, 2024)
ANIO_VALIDACION = 2024


def umbrales_por_mes(panel, hasta_anio):
    """Percentil 75 de la lluvia de cada mes calendario, usando solo anios <= hasta_anio."""
    hist = panel[panel["anio"] <= hasta_anio]
    return hist.groupby("mes")["lluvia_mm"].quantile(PERCENTIL / 100).to_dict()


def etiquetar(panel, umbrales):
    df = panel.copy()
    df[OBJETIVO] = [int(v > umbrales[m]) for v, m in zip(df["lluvia_mm"], df["mes"])]
    return df


def ajustar(X, y, balancear=False):
    """Sin pesos de clase las probabilidades salen calibradas; el recall se
    controla despues con el umbral, no falseando la escala de probabilidades."""
    ratio = y.sum() / max(len(y) - y.sum(), 1)
    return {
        "Regresion logistica": RegresionLogistica(alpha=1.0, peso_clase="balanceado" if balancear else "ninguno").fit(X, y),
        "XGBoost": _ensamble(ratio if balancear else 1.0).fit(X, y),
    }


def barrer(y, p):
    filas = []
    for u in sorted(set(np.round(np.quantile(p, np.linspace(0.0, 0.999, 200)), 6))):
        pred = (p >= u).astype(int)
        filas.append(
            {
                "umbral": float(u),
                "sensibilidad": recall_score(y, pred, zero_division=0),
                "precision": precision_score(y, pred, zero_division=0),
            }
        )
    return pd.DataFrame(filas)


def umbral_por_recall(curva, minimo=0.70):
    """Elige el umbral mas alto que aun alcance el recall minimo. Asi se
    maximiza la precision sin perder sensibilidad."""
    ok = curva[curva["sensibilidad"] >= minimo]
    if len(ok) == 0:
        return float(curva.iloc[0]["umbral"])
    return float(ok.sort_values("precision", ascending=False).iloc[0]["umbral"])


def split_temporal(panel, anio_validacion=ANIO_VALIDACION):
    u = umbrales_por_mes(panel, anio_validacion - 1)
    f = construir_features(etiquetar(panel, u))
    return f, u


def evaluar():
    crudo = pd.read_csv(RUTA_PANEL)
    f, umbrales = split_temporal(crudo)

    tr = f[f["anio"].isin(ANIO_ENTRENAMIENTO)]
    va = f[f["anio"] == ANIO_VALIDACION]

    modelos = ajustar(tr[FEATURES], tr[OBJETIVO].to_numpy())
    salida = {}
    for nombre, m in modelos.items():
        p_va = m.predict_proba(va[FEATURES])[:, 1]
        y_va = va[OBJETIVO].to_numpy()
        curva = barrer(y_va, p_va)
        u = umbral_por_recall(curva)
        pred = (p_va >= u).astype(int)
        salida[nombre] = {
            "modelo": m,
            "auc": roc_auc_score(y_va, p_va),
            "umbral": u,
            "curva": curva,
            "prob_media": float(p_va.mean()),
            "frecuencia_real": float(y_va.mean()),
            "sensibilidad": recall_score(y_va, pred, zero_division=0),
            "precision": precision_score(y_va, pred, zero_division=0),
            "tp": int(((pred == 1) & (y_va == 1)).sum()),
            "fp": int(((pred == 1) & (y_va == 0)).sum()),
            "fn": int(((pred == 0) & (y_va == 1)).sum()),
        }
    return {"panel": f, "entrena": tr, "valida": va, "resultados": salida, "umbrales": umbrales, "crudo": crudo}


def backtest_eneros(crudo, anios):
    """Backtest con ventana expansiva: para cada enero de `anios` se entrena
    solo con anios anteriores y se rankean los 87 municipios. Cada corte
    recalcula sus umbrales con los anos previos, sin mirar el enero evaluado."""
    filas = []
    for anio in anios:
        previos = crudo[crudo["anio"] < anio]
        f_prev = construir_features(previos)
        f_all, u = split_temporal(crudo, anio - 1)
        etiquetas_prev = etiquetar(f_prev, umbrales_por_mes(crudo, anio - 1))[OBJETIVO]
        f_prev = f_prev.copy()
        f_prev[OBJETIVO] = etiquetas_prev.to_numpy()
        f_prev = f_prev.merge(crudo[["codigo_dane", "anio", "mes", "municipio", "lluvia_mm"]], on=["codigo_dane", "anio", "mes"], how="left")

        enero_real = crudo[(crudo["anio"] == anio) & (crudo["mes"] == 1)].copy()
        enero_real[OBJETIVO] = [int(v > u[1]) for v in enero_real["lluvia_mm"]]

        X_prev = f_prev.dropna(subset=FEATURES + [OBJETIVO])
        modelos = ajustar(X_prev[FEATURES], X_prev[OBJETIVO].to_numpy())

        enero_feat = f_all[(f_all["anio"] == anio) & (f_all["mes"] == 1)].copy()
        if len(enero_feat) == 0:
            continue
        base = enero_real[OBJETIVO].mean()

        for nombre, m in modelos.items():
            p = m.predict_proba(enero_feat[FEATURES])[:, 1]
            t = enero_real.copy()
            t["prob"] = p
            top = t.nlargest(10, "prob")
            filas.append(
                {
                    "anio": anio,
                    "modelo": nombre,
                    "n_lluviosos": int(enero_real[OBJETIVO].sum()),
                    "top10_lluviosos": int(top[OBJETIVO].sum()),
                    "lift": float(top[OBJETIVO].sum() / (10 * base)) if base > 0 else np.nan,
                    "prob_top10_media": float(top["prob"].mean()),
                    "prob_real": float(base),
                    "auc": float(roc_auc_score(enero_real[OBJETIVO], p)) if enero_real[OBJETIVO].sum() > 0 else np.nan,
                }
            )
    return pd.DataFrame(filas)


def predecir_enero_2025():
    crudo = pd.read_csv(RUTA_PANEL)
    u = umbrales_por_mes(crudo, crudo["anio"].max())
    eti = etiquetar(crudo, u)
    f = construir_features(eti)

    ultimo = crudo.sort_values(["codigo_dane", "anio", "mes"]).groupby("codigo_dane").tail(1)
    filas = [
        {"codigo_dane": r.codigo_dane, "municipio": r.municipio, "anio": 2025, "mes": 1, "altitud_m": int(r.altitud_m)}
        for r in ultimo.itertuples()
    ]
    f = construir_features(_anexar_filas_futuras(eti, filas))
    enero = f[f["anio"] == 2025].copy()

    tr = f[f["anio"].isin(ANIO_ENTRENAMIENTO)]
    modelos = ajustar(tr[FEATURES], tr[OBJETIVO].to_numpy())

    salida = enero[["codigo_dane", "municipio", "anio", "mes", "altitud_m"]].reset_index(drop=True)
    for col in FEATURES:
        if col not in salida.columns:
            salida[col] = enero[col].to_numpy()
    salida["lluvia_umbral"] = u[1]

    for nombre, m in modelos.items():
        salida[f"prob_{nombre}"] = (100 * m.predict_proba(enero[FEATURES])[:, 1]).round(1)
    return salida, u, modelos


def linea_base_enero(crudo, anios):
    """Para enero de `anios`, predice si el mismo municipio fue lluvioso en
    enero del anio anterior."""
    filas = []
    for anio in anios:
        u = umbrales_por_mes(crudo, anio - 1)
        actual = crudo[(crudo["anio"] == anio) & (crudo["mes"] == 1)]
        prev = crudo[(crudo["anio"] == anio - 1) & (crudo["mes"] == 1)]
        j = actual.merge(prev[["codigo_dane", "lluvia_mm"]], on="codigo_dane", suffixes=("", "_prev"))
        y = (j["lluvia_mm"] > u[1]).astype(int)
        pred = (j["lluvia_mm_prev"] > u[1]).astype(int)
        filas.append(
            {
                "anio": anio,
                "pos": int(y.sum()),
                "tp": int(((y == 1) & (pred == 1)).sum()),
                "fp": int(((y == 0) & (pred == 1)).sum()),
                "fn": int(((y == 1) & (pred == 0)).sum()),
            }
        )
    return pd.DataFrame(filas)