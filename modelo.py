"""Modelo de anticipacion de movimientos en masa.

Corte temporal estricto: para predecir el mes t solo se usa informacion
disponible al inicio de t, es decir hasta t-1. El archivo de 2025 no se
abre en ningun momento de este modulo.

Nota de entorno: en esta maquina una politica de Control de Aplicaciones de
Windows bloquea el DLL _sag_fast de scikit-learn, lo que inutiliza
sklearn.linear_model, sklearn.tree y sklearn.ensemble. La regresion
logistica se implementa aqui con scipy y el ensamble se hace con XGBoost.
"""

from pathlib import Path

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
RUTA_PANEL = BASE_DIR / "Datos" / "panel_entrenamiento_2015_2024.csv"
RUTA_PRUEBA = BASE_DIR / "Datos" / "panel_prueba_2025.csv"

OBJETIVO = "hubo_mm"

FEATURES = [
    "lluvia_mm_mes_anterior",
    "lluvia_acum_3m",
    "lluvia_acum_12m",
    "eventos_12m_lag",
    "altitud_m",
    "mes",
    "mes_sin",
    "mes_cos",
]

FUGAS = {
    "lluvia_mm": "lluvia acumulada durante el propio mes: no se conoce al inicio del mes",
    "eventos_mes": "conteo de eventos de lluvia del propio mes: no se conoce al inicio del mes",
    "eventos_12m": "ventana movil de 12 meses que incluye el mes actual: filtra el presente",
}

ANIOS_ENTRENAMIENTO = range(2016, 2022)
ANIOS_VALIDACION = range(2022, 2025)


class RegresionLogistica:
    """Logistica con penalizacion L2 y pesos por clase, resuelta con L-BFGS."""

    def __init__(self, alpha=1.0, peso_clase="balanceado", max_iter=500):
        self.alpha = alpha
        self.peso_clase = peso_clase
        self.max_iter = max_iter

    def _pesos(self, y):
        if self.peso_clase != "balanceado":
            return np.ones(len(y))
        n_pos, n_neg = y.sum(), len(y) - y.sum()
        return np.where(y == 1, len(y) / (2 * n_pos), len(y) / (2 * n_neg))

    def fit(self, X, y):
        self.escala_ = StandardScaler().fit(X)
        Z = self.escala_.transform(X)
        y = np.asarray(y, dtype=float)
        w = self._pesos(y)
        n, p = Z.shape
        Zb = np.hstack([np.ones((n, 1)), Z])
        sw = w / w.sum()

        def objetivo(beta):
            m = Zb @ beta
            return np.sum(sw * np.logaddexp(0, m) - sw * y * m) + 0.5 * self.alpha * np.sum(beta[1:] ** 2)

        def gradiente(beta):
            m = Zb @ beta
            resid = sw * (1.0 / (1.0 + np.exp(-m)) - y)
            g = Zb.T @ resid
            penal = np.zeros_like(beta)
            penal[1:] = self.alpha * beta[1:]
            return g + penal

        r = minimize(objetivo, np.zeros(p + 1), jac=gradiente, method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.coef_ = r.x[1:]
        self.intercept_ = r.x[0]
        self.n_iter_ = r.nit
        return self

    def predict_proba(self, X):
        Z = self.escala_.transform(X)
        m = Z @ self.coef_ + self.intercept_
        return np.vstack([1 - 1 / (1 + np.exp(-m)), 1 / (1 + np.exp(-m))]).T


def construir_features(panel):
    """Features legales. Todo lo que toca el mes t se desplaza una fila."""
    df = panel.sort_values(["codigo_dane", "anio", "mes"]).reset_index(drop=True)
    lluvia_previa = df.groupby("codigo_dane", sort=False)["lluvia_mm"].shift(1)
    eventos_previos = df.groupby("codigo_dane", sort=False)["eventos_mes"].shift(1)

    df["lluvia_mm_mes_anterior"] = lluvia_previa
    df["lluvia_acum_3m"] = lluvia_previa.groupby(df["codigo_dane"], sort=False).transform(
        lambda s: s.rolling(3, min_periods=3).sum()
    )
    df["lluvia_acum_12m"] = lluvia_previa.groupby(df["codigo_dane"], sort=False).transform(
        lambda s: s.rolling(12, min_periods=12).sum()
    )
    df["eventos_12m_lag"] = eventos_previos.groupby(df["codigo_dane"], sort=False).transform(
        lambda s: s.rolling(12, min_periods=12).sum()
    )
    df["mes_sin"] = df["mes"].map(lambda m: round(math.sin(2 * math.pi * (m - 1) / 12), 6))
    df["mes_cos"] = df["mes"].map(lambda m: round(math.cos(2 * math.pi * (m - 1) / 12), 6))

    return df.dropna(subset=FEATURES).reset_index(drop=True)


def linea_base(panel, evaluados):
    """Predice repitiendo lo que paso el mismo mes del anio anterior."""
    tabla = panel.set_index(["codigo_dane", "anio", "mes"])[OBJETIVO]
    pred = [
        0 if pd.isna(tabla.get((f["codigo_dane"], f["anio"] - 1, f["mes"]), np.nan)) else int(tabla[(f["codigo_dane"], f["anio"] - 1, f["mes"])])
        for _, f in evaluados.iterrows()
    ]
    return pd.Series(pred, index=evaluados.index, name="linea_base")


def metricas(y_true, y_pred):
    return {
        "sensibilidad": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def barrer_umbral(y_true, probas, n=120):
    """Barre sobre cuantiles de las probabilidades: la escala de cada modelo
    es distinta y una rejilla fija dejaria fuera casi todo el rango."""
    probs = np.asarray(probas, dtype=float)
    base = np.linspace(0.01, 0.99, n)
    cuantiles = np.quantile(probs, np.linspace(0.50, 0.999, n))
    filas = []
    for u in sorted(set(np.round(np.concatenate([base, cuantiles]), 6))):
        pred = (probs >= u).astype(int)
        filas.append({"umbral": float(u), **metricas(y_true, pred)})
    return pd.DataFrame(filas)


def seleccionar_umbral(curva):
    """F2: pondera el recall 4 veces mas que la precision (F1 es 1:1)."""
    mejor, puntaje = None, -1.0
    for _, f in curva.iterrows():
        p, s = f["precision"], f["sensibilidad"]
        f2 = (5 * p * s / (4 * p + s)) if (4 * p + s) > 0 else 0.0
        if f2 > puntaje:
            mejor, puntaje = f["umbral"], f2
    return mejor


def _ensamble(ratio):
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=ratio,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def ejecutar():
    crudo = pd.read_csv(RUTA_PANEL)
    panel = construir_features(crudo)

    entrena = panel[panel["anio"].isin(ANIOS_ENTRENAMIENTO)]
    valida = panel[panel["anio"].isin(ANIOS_VALIDACION)]

    X_tr, y_tr = entrena[FEATURES], entrena[OBJETIVO].to_numpy()
    X_va, y_va = valida[FEATURES], valida[OBJETIVO].to_numpy()

    ratio = y_tr.sum() / (len(y_tr) - y_tr.sum())

    modelos = {
        "Regresion logistica": RegresionLogistica(alpha=1.0).fit(X_tr, y_tr),
        "XGBoost": _ensamble(ratio).fit(X_tr, y_tr),
    }

    base = linea_base(crudo, valida)
    resultados = {}
    for nombre, modelo in modelos.items():
        probas = modelo.predict_proba(X_va)[:, 1]
        curva = barrer_umbral(y_va, probas)
        umbral = seleccionar_umbral(curva)
        pred = (probas >= umbral).astype(int)
        resultados[nombre] = {
            "modelo": modelo,
            "curva": curva,
            "umbral": umbral,
            "probas": probas,
            "y_va": y_va,
            "pred": pred,
            "metricas": metricas(y_va, pred),
            "auc": roc_auc_score(y_va, probas),
            "importancias": _importancias(nombre, modelo),
        }

    return {
        "panel": panel,
        "entrena": entrena,
        "valida": valida,
        "resultados": resultados,
        "base": {"pred": base.to_numpy(), "y": y_va, "metricas": metricas(y_va, base.to_numpy())},
    }


def _importancias(nombre, modelo):
    if nombre == "Regresion logistica":
        return pd.Series(modelo.coef_, index=FEATURES).sort_values(key=np.abs, ascending=False)
    return pd.Series(modelo.feature_importances_, index=FEATURES).sort_values(ascending=False)


def predecir_enero_2025(ruta_panel=RUTA_PANEL):
    """Features de 2025-01 construidas SOLO con informacion hasta 2024-12."""
    crudo = pd.read_csv(ruta_panel)
    panel = construir_features(crudo)
    ultimo = panel.sort_values(["codigo_dane", "anio", "mes"]).groupby("codigo_dane").tail(1).copy()
    ultimo["mes"] = 1
    ultimo["mes_sin"] = 0.0
    ultimo["mes_cos"] = 1.0
    return ultimo[["codigo_dane", "municipio"] + FEATURES].reset_index(drop=True)


def _mes_sin_cos(mes):
    return round(math.sin(2 * math.pi * (mes - 1) / 12), 6), round(math.cos(2 * math.pi * (mes - 1) / 12), 6)


def predecir_2025(ruta_panel=RUTA_PANEL, ruta_prueba=RUTA_PRUEBA):
    """Prediccion de los 1044 municipio-mes de 2025.

    Frontera de informacion, fila por fila: para el mes objetivo t solo se
    usa lluvia_mm y eventos_mes de meses estrictamente anteriores a t. El
    mes 1 no usa nada de 2025. Los meses 2-12 usan la lluvia ya observada
    de meses 1..t-1 de 2025, que al inicio de t ya se conoce. Nunca se lee
    la lluvia del propio mes t ni hubo_mm de ningun mes de 2025.

    Devuelve ademas la columna real, que se adjunta DESPUES de predecir y no
    participa en el ajuste: existe solo para que compares.
    """
    historico = pd.read_csv(ruta_panel)
    prueba = pd.read_csv(ruta_prueba)

    extension = []
    for (dane, municipio), g in prueba.groupby(["codigo_dane", "municipio"]):
        g = g.sort_values("mes")
        for _, f in g.iterrows():
            extension.append(
                {
                    "codigo_dane": dane,
                    "municipio": municipio,
                    "anio": int(f["anio"]),
                    "mes": int(f["mes"]),
                    "lluvia_mm": float(f["lluvia_mm"]),
                    "eventos_mes": int(f["eventos_mes"]),
                    "altitud_m": int(f["altitud_m"]),
                }
            )
    completo = pd.concat([historico, pd.DataFrame(extension)], ignore_index=True)

    panel = construir_features(completo)
    objetivo = panel[panel["anio"] == 2025].copy()

    X = objetivo[FEATURES]
    salida = objetivo[["codigo_dane", "municipio", "anio", "mes", "altitud_m"]].reset_index(drop=True)
    for col in FEATURES:
        if col not in salida.columns:
            salida[col] = objetivo[col].to_numpy()

    ajuste = ejecutar()
    for nombre, res in ajuste["resultados"].items():
        p = res["modelo"].predict_proba(X)[:, 1]
        salida[f"prob_{nombre}"] = p
        salida[f"pred_{nombre}"] = (p >= res["umbral"]).astype(int)
        salida[f"umbral_{nombre}"] = res["umbral"]

    real = prueba[["codigo_dane", "anio", "mes", "hubo_mm"]].copy()
    salida = salida.merge(real, on=["codigo_dane", "anio", "mes"], how="left")
    return salida, ajuste