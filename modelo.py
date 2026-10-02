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

UMBRAL_LLUVIA = 150.0
OBJETIVO_LLUVIA = "lluvia_intensa"


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


def ejecutar_lluvia():
    """Probabilidad de que el mes supere UMBRAL_LLUVIA mm.

    No es un pronostico meteorologico: con este archivo no hay con que
    anticipar el clima de un mes futuro. Lo que se estima es la probabilidad
    climatologica de lluvia intensa condicionada a lo que ya ocurrio en los
    meses anteriores. Usa las mismas features legales del modelo de
    movimientos en masa: para el mes t solo entra informacion hasta t-1.
    """
    crudo = pd.read_csv(RUTA_PANEL).sort_values(["codigo_dane", "anio", "mes"]).reset_index(drop=True)
    crudo[OBJETIVO_LLUVIA] = (crudo["lluvia_mm"] > UMBRAL_LLUVIA).astype(int)

    panel = construir_features(crudo)

    entrena = panel[panel["anio"].isin(ANIOS_ENTRENAMIENTO)]
    valida = panel[panel["anio"].isin(ANIOS_VALIDACION)]

    X_tr, y_tr = entrena[FEATURES], entrena[OBJETIVO_LLUVIA].to_numpy()
    X_va, y_va = valida[FEATURES], valida[OBJETIVO_LLUVIA].to_numpy()

    ratio = y_tr.sum() / (len(y_tr) - y_tr.sum())
    base_va = valida["lluvia_mm"].to_numpy() > UMBRAL_LLUVIA

    modelos = {
        "Regresion logistica": RegresionLogistica(alpha=1.0).fit(X_tr, y_tr),
        "XGBoost": _ensamble(ratio).fit(X_tr, y_tr),
    }

    resultados = {}
    for nombre, m in modelos.items():
        p = m.predict_proba(X_va)[:, 1]
        curva = barrer_umbral(y_va, p)
        umbral = seleccionar_umbral(curva)
        resultados[nombre] = {
            "modelo": m,
            "curva": curva,
            "umbral": umbral,
            "probas": p,
            "metricas": metricas(y_va, (p >= 0.5).astype(int)),
            "metricas_umbral": metricas(y_va, (p >= umbral).astype(int)),
            "auc": roc_auc_score(y_va, p),
            "y_va": y_va,
        }

    return {
        "panel": panel,
        "entrena": entrena,
        "valida": valida,
        "resultados": resultados,
        "base": metricas(y_va, base_va),
        "umbral": UMBRAL_LLUVIA,
    }


def predecir_lluvia_enero_2025():
    """Enero 2025 con la misma frontera de informacion: features hasta dic 2024."""
    historico = pd.read_csv(RUTA_PANEL)
    ultimo = historico.sort_values(["codigo_dane", "anio", "mes"]).groupby("codigo_dane").tail(1)
    filas = [
        {"codigo_dane": f["codigo_dane"], "municipio": f["municipio"], "anio": 2025, "mes": 1, "altitud_m": int(f["altitud_m"])}
        for _, f in ultimo.iterrows()
    ]
    panel = construir_features(_anexar_filas_futuras(historico, filas))
    enero = panel[panel["anio"] == 2025].copy()

    X = enero[FEATURES]
    salida = enero[["codigo_dane", "municipio", "anio", "mes", "altitud_m"]].reset_index(drop=True)
    for col in FEATURES:
        if col not in salida.columns:
            salida[col] = enero[col].to_numpy()

    ajuste = ejecutar_lluvia()
    for nombre, res in ajuste["resultados"].items():
        p = res["modelo"].predict_proba(X)[:, 1]
        salida[f"prob_{nombre}"] = (100 * p).round(1)
        salida[f"pred_{nombre}"] = (p >= res["umbral"]).astype(int)
        salida[f"umbral_{nombre}"] = res["umbral"]
    return salida, ajuste


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


def predecir_enero_2025(ruta_panel=RUTA_PANEL, ruta_prueba=RUTA_PRUEBA):
    """Enero 2025 con la misma ruta que el resto de meses: la lluvia de
    diciembre 2024 entra como rezago. Delegar en predecir_2025 evita que
    las dos rutas se desincronicen."""
    tabla, _, _ = predecir_2025(ruta_panel, ruta_prueba)
    quitar = [c for c in tabla.columns if c.startswith(("prob_", "pred_", "umbral_")) or c == "hubo_mm"]
    return tabla[tabla["mes"] == 1].drop(columns=quitar).reset_index(drop=True)


def _mes_sin_cos(mes):
    return round(math.sin(2 * math.pi * (mes - 1) / 12), 6), round(math.cos(2 * math.pi * (mes - 1) / 12), 6)


def _anexar_filas_futuras(historico, filas):
    """Agrega filas de meses futuros con lluvia unknown como NaN. Como todas las
    features se construyen con shift(1), una fila NaN solo afecta a los meses
    posteriores a ella, nunca a la suya propia."""
    vacias = historico.iloc[:0].copy()
    nuevas = []
    for f in filas:
        fila = vacias.iloc[0].copy() if len(vacias) else None
        registro = {
            "codigo_dane": f["codigo_dane"],
            "municipio": f["municipio"],
            "anio": int(f["anio"]),
            "mes": int(f["mes"]),
            "lluvia_mm": f.get("lluvia_mm", np.nan),
            "eventos_mes": f.get("eventos_mes", np.nan),
            "altitud_m": f["altitud_m"],
        }
        nuevas.append(registro)
    return pd.concat([historico, pd.DataFrame(nuevas)], ignore_index=True)


def predecir_2025(ruta_panel=RUTA_PANEL, ruta_prueba=RUTA_PRUEBA):
    """Prediccion de 2025 sin usar el resultado real de 2025.

    Si el archivo de prueba esta disponible se construye la serie completa de
    12 meses de forma secuencial: para el mes objetivo t solo entra la lluvia
    de meses anteriores a t. Si no esta disponible, el unico mes que se puede
    anticipar sin datos de 2025 es enero, porque sus features provienen de
    2024.

    Nunca se lee hubo_mm de 2025 para entrenar, validar ni elegir umbral.
    """
    historico = pd.read_csv(ruta_panel)
    existe_prueba = Path(ruta_prueba).exists()
    prueba = pd.read_csv(ruta_prueba) if existe_prueba else None

    if existe_prueba:
        filas = [
            {
                "codigo_dane": dane,
                "municipio": municipio,
                "anio": int(f["anio"]),
                "mes": int(f["mes"]),
                "lluvia_mm": float(f["lluvia_mm"]),
                "eventos_mes": int(f["eventos_mes"]),
                "altitud_m": int(f["altitud_m"]),
            }
            for (dane, municipio), g in prueba.groupby(["codigo_dane", "municipio"])
            for _, f in g.sort_values("mes").iterrows()
        ]
    else:
        ultimo = historico.sort_values(["codigo_dane", "anio", "mes"]).groupby("codigo_dane").tail(1)
        filas = [
            {
                "codigo_dane": f["codigo_dane"],
                "municipio": f["municipio"],
                "anio": 2025,
                "mes": 1,
                "altitud_m": int(f["altitud_m"]),
            }
            for _, f in ultimo.iterrows()
        ]

    panel = construir_features(_anexar_filas_futuras(historico, filas))
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

    if existe_prueba:
        real = prueba[["codigo_dane", "anio", "mes", "hubo_mm"]].copy()
        salida = salida.merge(real, on=["codigo_dane", "anio", "mes"], how="left")

    return salida, ajuste, existe_prueba