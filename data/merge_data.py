import os, re
import pandas as pd

# === RUTAS ===
DATA_DIR = os.path.join("datos", "vacas")
OUT_DIR  = "out"
OUT_CSV  = os.path.join(OUT_DIR, "registros_sesiones_merged.csv")

os.makedirs(OUT_DIR, exist_ok=True)

def read_loose(path):
    """Lee CSV con autodetección de separador y codificación, como texto."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, header=None, dtype=str)
        except Exception:
            continue
    # ultimo intento: coma fija
    return pd.read_csv(path, encoding="utf-8", header=None, dtype=str)

def sanitize(s):
    """Devuelve texto limpio; si viene NaN/None/float vacío, regresa ''. """
    try:
        if s is None:
            return ""
        if isinstance(s, float):
            if pd.isna(s):
                return ""
        s = str(s)
        s = re.sub(r"\s+", " ", s.strip())
        return s
    except Exception:
        return ""

def ffill_row(values):
    """Forward-fill en la fila de grupos, tolerante a vacíos/NaN."""
    out, last = [], ""
    for v in values:
        v = sanitize(v)
        if v != "":
            last = v
        out.append(last)
    return out

def make_unique(names):
    seen, out = {}, []
    for n in names:
        base = n if n else "col"
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__{seen[base]}")
    return out

def read_with_grouped_header(path):
    """
    Fila 0: GRUPO (Main, Estado, Conductividad …)  [con forward-fill]
    Fila 1: SUBCOLUMNA (Hora de inicio, Acción, DI, DD, TI, TD, …)
    Nombres: '<Grupo> | <Sub>' (o '<Sub>' si no hay grupo)
    Devuelve data desde fila 2 hacia abajo, sin transformar valores.
    """
    df_raw = read_loose(path)
    if len(df_raw) < 2:
        return pd.DataFrame()

    groups = ffill_row(list(df_raw.iloc[0].values))
    subs   = [sanitize(x) for x in list(df_raw.iloc[1].values)]

    cols = []
    for g, s in zip(groups, subs):
        if g and s:
            cols.append(f"{g} | {s}")
        elif s:
            cols.append(s)
        elif g:
            cols.append(g)
        else:
            cols.append("")

    cols = make_unique(cols)
    body = df_raw.iloc[2:].copy()
    body.columns = cols
    body = body.dropna(axis=1, how="all")
    return body

def main():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"No se encontró el directorio: {DATA_DIR}")

    files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")])
    if not files:
        raise SystemExit(f"No hay archivos CSV en {DATA_DIR}")

    frames = []
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        df = read_with_grouped_header(path)
        if df.empty:
            print(f"⚠️ {fname}: sin datos válidos (¿archivo vacío o solo encabezados?)")
            continue

        cow_id = re.sub(r"\D", "", os.path.splitext(fname)[0]) or None
        df.insert(0, "id", cow_id)
        frames.append(df)
        print(f"✓ {fname}: {len(df)} filas, {len(df.columns)} columnas")

    if not frames:
        raise SystemExit("No se generaron datos válidos.")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"\n🚀 Listo: {OUT_CSV}")
    print(f"Filas totales: {len(merged)} | Columnas totales: {len(merged.columns)}")
    print("Algunas columnas:", list(merged.columns)[:12])

if __name__ == "__main__":
    main()
