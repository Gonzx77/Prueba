from pathlib import Path

import json

import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
RUTA_PANEL = BASE_DIR / "Datos" / "panel_entrenamiento_2015_2024.csv"
RUTA_GEOJSON = BASE_DIR / "Datos" / "santander_municipios.geojson"

CLAVE = ["codigo_dane", "anio", "mes"]
POR_PAGINA = 50

app = Flask(__name__)
_cache = {}


def _paso(nombre, estado, detalle, filas=0):
    return {"nombre": nombre, "estado": estado, "detalle": detalle, "filas": filas}


def _normalizar_texto(serie):
    return serie.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)


def limpiar_panel(df):
    reporte = [_paso("Carga", "ok", f"{len(df):,} filas x {df.shape[1]} columnas leidas del CSV")]

    df = df.copy()
    df["municipio"] = _normalizar_texto(df["municipio"])
    n = int(df["municipio"].isna().sum() + (df["municipio"].fillna("") == "").sum())
    reporte.append(
        _paso(
            "Espacios y titulos",
            "ok" if n == 0 else "aviso",
            f"{n} nombres con espacios sobrantes o vacios",
            n,
        )
    )

    tipos = {
        "codigo_dane": "int64",
        "anio": "int64",
        "mes": "int64",
        "lluvia_mm": "float64",
        "lluvia_mm_mes_anterior": "float64",
        "eventos_mes": "int64",
        "hubo_mm": "int64",
        "eventos_12m": "int64",
        "altitud_m": "int64",
    }
    convertidos = 0
    for col, dtype in tipos.items():
        antes = df[col].dtype
        df[col] = pd.to_numeric(df[col], errors="coerce")
        convertidos += int(df[col].isna().sum())
        if antes != dtype:
            df[col] = df[col].astype(dtype)
    reporte.append(
        _paso(
            "Tipos de dato",
            "ok" if convertidos == 0 else "aviso",
            f"{len(tipos)} columnas tipadas, {convertidos} valores no numericos convertidos a nulo",
            convertidos,
        )
    )

    dups_exactos = int(df.duplicated().sum())
    df = df.drop_duplicates()
    reporte.append(_paso("Duplicados exactos", "ok", f"{dups_exactos} filas repetidas eliminadas", dups_exactos))

    dups_clave = int(df.duplicated(subset=CLAVE).sum())
    if dups_clave:
        df = df.sort_values(CLAVE).drop_duplicates(subset=CLAVE, keep="last")
    reporte.append(
        _paso("Duplicados por llave", "ok", f"{dups_clave} filas con llave repetida (codigo_dane+anio+mes)", dups_clave)
    )

    nulos = df.isna().sum()
    total_nulos = int(nulos.sum())
    reporte.append(
        _paso(
            "Valores nulos",
            "ok" if total_nulos == 0 else "aviso",
            "0 nulos" if total_nulos == 0 else f"nulos por columna: {nulos[nulos > 0].to_dict()}",
            total_nulos,
        )
    )

    mes_malo = int((~df["mes"].between(1, 12)).sum())
    anio_malo = int((~df["anio"].between(2015, 2024)).sum())
    if mes_malo:
        df = df[df["mes"].between(1, 12)]
    if anio_malo:
        df = df[df["anio"].between(2015, 2024)]
    reporte.append(
        _paso("Rangos de fecha", "ok", f"mes fuera de 1-12: {mes_malo} | anio fuera de 2015-2024: {anio_malo}", mes_malo + anio_malo)
    )

    df = df.sort_values(CLAVE).reset_index(drop=True)

    esperado = df["codigo_dane"].nunique() * df["anio"].nunique() * 12
    faltantes = esperado - len(df)
    reporte.append(
        _paso(
            "Panel balanceado",
            "ok" if faltantes == 0 else "aviso",
            f"{df['codigo_dane'].nunique()} municipios x {df['anio'].nunique()} anios x 12 meses = {esperado:,} esperado",
            max(faltantes, 0),
        )
    )

    esperado_lag = df.groupby("codigo_dane")["lluvia_mm"].shift(1)
    mask_lag = df["mes"] > 1
    mal_lag = int((~((df["lluvia_mm_mes_anterior"] - esperado_lag).abs().fillna(0) < 1e-9)[mask_lag]).sum())
    reporte.append(
        _paso(
            "Coherencia del rezago 1",
            "ok" if mal_lag == 0 else "aviso",
            f"{mal_lag} filas donde lluvia_mm_mes_anterior no coincide con el mes previo",
            mal_lag,
        )
    )

    calc_12m = df.groupby("codigo_dane")["eventos_mes"].transform(lambda s: s.rolling(12, min_periods=1).sum())
    dif = int((calc_12m != df["eventos_12m"]).sum())
    imposible = int((df["eventos_12m"] < df["eventos_mes"]).sum())
    reporte.append(
        _paso(
            "eventos_12m (ventana movil)",
            "sin tocar",
            f"{dif} filas no coinciden con la suma movil de eventos_mes, de las cuales {imposible} "
            "tienen eventos_12m < eventos_mes (imposible si la ventana incluye el mes actual). "
            "No se corrige: el modelo reconstruye su propia version sin fuga (eventos_12m_lag)",
            imposible,
        )
    )

    desalineado = int(((df["eventos_mes"] > 0) & (df["hubo_mm"] == 0)).sum())
    reporte.append(
        _paso(
            "eventos_mes y hubo_mm",
            "ok",
            f"{desalineado} filas con eventos_mes > 0 y hubo_mm = 0. No es un error: eventos_mes cuenta "
            "eventos de lluvia y hubo_mm indica si hubo movimiento en masa. Son cosas distintas",
            0,
        )
    )

    df["fecha"] = pd.to_datetime(dict(year=df["anio"], month=df["mes"], day=1))
    df["trimestre"] = df["mes"].sub(1).floordiv(3) + 1
    reporte.append(_paso("Variables derivadas", "ok", "fecha y trimestre agregadas (sin variables que filtren el futuro)"))

    return df, reporte


def cargar_datos(forzar=False):
    if forzar or "panel" not in _cache:
        crudo = pd.read_csv(RUTA_PANEL)
        panel, reporte = limpiar_panel(crudo)

        crudo_geo = json.loads(RUTA_GEOJSON.read_text(encoding="utf-8-sig"))
        geo = pd.DataFrame(
            [{"codigo_dane": int(f["properties"]["codigo_dane"]), "geometry": f["geometry"]} for f in crudo_geo["features"]]
        )
        sin_geo = sorted(set(panel["codigo_dane"]) - set(geo["codigo_dane"]))
        panel = panel.merge(geo, on="codigo_dane", how="left")

        _cache["panel"] = panel
        _cache["reporte"] = reporte
        _cache["sin_geo"] = sin_geo
    return _cache["panel"], _cache["reporte"], _cache["sin_geo"]


def _resumen(panel):
    return {
        "filas": f"{len(panel):,}",
        "municipios": panel["codigo_dane"].nunique(),
        "anios": f"{panel['anio'].min()}-{panel['anio'].max()}",
        "meses": f"{len(panel) / max(panel['codigo_dane'].nunique(), 1):,.0f} por municipio-anio",
        "lluvia_media": round(panel["lluvia_mm"].mean(), 1),
        "lluvia_max": round(panel["lluvia_mm"].max(), 1),
        "eventos_totales": int(panel["eventos_mes"].sum()),
        "meses_con_evento": int((panel["eventos_mes"] > 0).sum()),
        "tasa_evento": f"{100 * (panel['eventos_mes'] > 0).mean():.1f}%",
    }


@app.route("/")
def index():
    panel, reporte, sin_geo = cargar_datos(forzar=request.args.get("recargar") == "1")

    municipios = sorted(panel["municipio"].unique())
    municipio_sel = request.args.get("municipio")
    anio_sel = request.args.get("anio", type=int)

    vista = panel
    if municipio_sel:
        vista = vista[vista["municipio"] == municipio_sel]
    if anio_sel:
        vista = vista[vista["anio"] == anio_sel]

    pagina = max(request.args.get("pagina", 1, type=int), 1)
    total_paginas = max(-(-len(vista) // POR_PAGINA), 1)
    pagina = min(pagina, total_paginas)
    bloque = vista.iloc[(pagina - 1) * POR_PAGINA : pagina * POR_PAGINA]

    columnas = ["codigo_dane", "municipio", "anio", "mes", "lluvia_mm", "lluvia_mm_mes_anterior", "eventos_mes", "hubo_mm", "eventos_12m", "altitud_m"]
    tabla = bloque[columnas].to_html(index=False, classes="tabla", border=0, float_format=lambda v: f"{v:g}")

    return render_template(
        "index.html",
        titulo="Panel de precipitacion extrema - Santander",
        reporte=reporte,
        resumen=_resumen(panel),
        tabla=tabla,
        columnas=columnas,
        municipios=municipios,
        anios=sorted(panel["anio"].unique(), reverse=True),
        municipio_sel=municipio_sel,
        anio_sel=anio_sel,
        pagina=pagina,
        total_paginas=total_paginas,
        filas_vista=len(vista),
        filas_total=len(panel),
        inicio=(pagina - 1) * POR_PAGINA + 1,
        fin=min(pagina * POR_PAGINA, len(vista)),
        sin_geo=sin_geo,
    )


@app.route("/modelo")
def vista_modelo():
    import modelo as M

    if "ajuste" not in _cache:
        _cache["ajuste"] = M.ejecutar()
    r = _cache["ajuste"]

    base = r["base"]["metricas"]
    filas_modelos = []
    for nombre, res in r["resultados"].items():
        m = res["metricas"]
        curva = res["curva"]
        filas_modelos.append(
            {
                "nombre": nombre,
                "umbral": res["umbral"],
                "auc": res["auc"],
                "sensibilidad": m["sensibilidad"],
                "precision": m["precision"],
                "f1": m["f1"],
                "supera": m["sensibilidad"] > base["sensibilidad"],
                "importancias": res["importancias"].head(6),
                "curva": curva,
            }
        )

    principal = max(r["resultados"].items(), key=lambda kv: kv[1]["metricas"]["f1"])
    X = M.predecir_enero_2025()
    ranking = X.assign(probabilidad=principal[1]["modelo"].predict_proba(X[M.FEATURES])[:, 1]).sort_values(
        "probabilidad", ascending=False
    )
    ranking["probabilidad"] = (100 * ranking["probabilidad"]).round(1)
    umbral_pct = 100 * principal[1]["umbral"]
    alertas = ranking[ranking["probabilidad"] >= umbral_pct]

    return render_template(
        "modelo.html",
        titulo="Modelo de anticipacion de movimientos en masa",
        base=base,
        modelos=filas_modelos,
        principal=principal[0],
        ranking=ranking.head(10),
        umbral_pct=umbral_pct,
        n_alertas=len(alertas),
        n_municipios=len(ranking),
        fugas=M.FUGAS,
        features=M.FEATURES,
        n_tr=len(r["entrena"]),
        n_va=len(r["valida"]),
        pos_tr=int(r["entrena"]["hubo_mm"].sum()),
        pos_va=int(r["valida"]["hubo_mm"].sum()),
        anio_tr=f"{min(M.ANIOS_ENTRENAMIENTO)}&ndash;{max(M.ANIOS_ENTRENAMIENTO)}",
        anio_va=f"{min(M.ANIOS_VALIDACION)}&ndash;{max(M.ANIOS_VALIDACION)}",
    )


@app.route("/predicciones")
def vista_predicciones():
    import modelo as M

    if "predicciones" not in _cache:
        tabla, _ = M.predecir_2025()
        _cache["predicciones"] = tabla

    t = _cache["predicciones"]
    modelos = ["Regresion logistica", "XGBoost"]

    tabla_html = (
        t.sort_values(["anio", "mes", "codigo_dane"])
        .to_html(index=False, classes="tabla", border=0, float_format=lambda v: f"{v:g}")
    )

    resumen_meses = []
    for mes in range(1, 13):
        bloque = t[t["mes"] == mes]
        fila = {"mes": mes, "reales": int(bloque["hubo_mm"].sum()), "celdas": []}
        for n in modelos:
            pred = bloque[f"pred_{n}"]
            tp = int(((bloque["hubo_mm"] == 1) & (pred == 1)).sum())
            fn = int(((bloque["hubo_mm"] == 1) & (pred == 0)).sum())
            fp = int(((bloque["hubo_mm"] == 0) & (pred == 1)).sum())
            fila["celdas"].append(
                {
                    "texto": f"{tp / (tp + fn):.2f} / {tp / (tp + fp):.2f}" if tp + fn and tp + fp else "s/p",
                    "detalle": f"TP {tp} · FN {fn} · FP {fp}",
                }
            )
        resumen_meses.append(fila)

    return render_template(
        "predicciones.html",
        titulo="Predicciones 2025",
        tabla=tabla_html,
        modelos=modelos,
        resumen_meses=resumen_meses,
        n=len(t),
        anio=2025,
    )


@app.route("/descargar-predicciones")
def descargar_predicciones():
    import modelo as M

    if "predicciones" not in _cache:
        _cache["predicciones"], _ = M.predecir_2025()
    csv = _cache["predicciones"].to_csv(index=False).encode("utf-8")
    return csv, 200, {
        "Content-Type": "text/csv; charset=utf-8",
        "Content-Disposition": "attachment; filename=predicciones_2025.csv",
    }


@app.route("/descargar")
def descargar():
    panel, _, _ = cargar_datos()
    columnas = ["codigo_dane", "municipio", "anio", "mes", "lluvia_mm", "lluvia_mm_mes_anterior", "eventos_mes", "hubo_mm", "eventos_12m", "altitud_m"]
    return panel[columnas].to_csv(index=False).encode("utf-8"), 200, {
        "Content-Type": "text/csv; charset=utf-8",
        "Content-Disposition": "attachment; filename=panel_limpio.csv",
    }


if __name__ == "__main__":
    app.run(debug=True, port=5000)