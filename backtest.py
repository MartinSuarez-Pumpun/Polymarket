"""
backtest.py
===========
Analiza mercados resueltos en el dataset y compara lo que
predecía el sistema contra el outcome real.

Genera un informe HTML visual con:
- Accuracy del modelo por bucket (SEGURA/CINCUENTA/NO_NO)
- Calibración: cuando decía X%, ¿acertó?
- Los mejores y peores casos
- Tabla completa de mercados resueltos

Uso:
    python backtest.py
    # Abre backtest_report.html en el navegador
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import config

log = config.get_logger("backtest")


def load_model():
    """Carga modelo y scaler."""
    xgb_path    = Path(config.MODEL_DIR) / "model.xgb.json"
    scaler_path = Path(config.SCALER_FILE)

    if not xgb_path.exists():
        log.warning("No hay modelo entrenado — usando heurístico")
        return None, None

    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(str(xgb_path))

    scaler = None
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    return model, scaler


def predict_proba(model, scaler, records):
    """Predice probabilidades para una lista de records."""
    if model is None:
        # Fallback heurístico
        from collector import heuristic_score
        return [heuristic_score(r) / 100.0 for r in records]

    X = np.array([[r.get(c, 0) for c in config.FEATURE_COLS] for r in records], dtype=np.float32)
    if scaler:
        X = scaler.transform(X)
    return model.predict_proba(X)[:, 1].tolist()


def classify(prob):
    if prob >= 0.68:   return "SEGURA"
    elif prob >= 0.42: return "CINCUENTA"
    else:              return "NO_NO"


def run_backtest():
    path = Path(config.RAW_DATA_FILE)
    if not path.exists():
        log.error("No existe el dataset")
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                pass

    # Solo mercados resueltos con label real
    resolved = [r for r in records if r.get("resolved") and r.get("has_real_label")]
    log.info(f"Dataset: {len(records)} total | {len(resolved)} resueltos con label real")

    if not resolved:
        log.warning("No hay mercados resueltos con label real todavía.")
        log.warning("El sistema necesita acumular snapshots durante 2-4 semanas.")
        log.warning("Por ahora el backtest usará los labels heurísticos.")
        # Fallback: usar todos los que tengan label_good definido
        resolved = [r for r in records if r.get("label_good") is not None]
        log.info(f"Usando {len(resolved)} registros con label heurístico")

    if not resolved:
        return []

    model, scaler = load_model()
    probas        = predict_proba(model, scaler, resolved)

    results = []
    for r, prob in zip(resolved, probas):
        bucket       = classify(prob)
        label_actual = r.get("label_good", 0)  # 1=bueno, 0=malo
        resolution   = r.get("resolution_price")  # 1.0=YES, 0.0=NO, None=desconocido

        # ¿Acertó el modelo?
        # SEGURA → esperaba que fuera "bueno" → correcto si label=1
        # NO_NO  → esperaba que fuera "malo"  → correcto si label=0
        # CINCUENTA → zona incierta
        if bucket == "SEGURA":
            correct = label_actual == 1
        elif bucket == "NO_NO":
            correct = label_actual == 0
        else:
            correct = None  # inconclusivo

        results.append({
            "question"      : r.get("question", "")[:80],
            "slug"          : r.get("slug", ""),
            "category"      : r.get("category_raw", "unknown"),
            "prob"          : round(prob, 3),
            "bucket"        : bucket,
            "label_actual"  : label_actual,
            "resolution"    : resolution,
            "has_real_label": r.get("has_real_label", False),
            "correct"       : correct,
            "price_yes"     : r.get("price_yes", 0.5),
            "price_at_res"  : r.get("price_yes_at_resolution"),
            "volume_24h"    : r.get("volume_24h", 0),
            "liquidity"     : r.get("liquidity", 0),
            "collected_at"  : r.get("collected_at", "")[:10],
        })

    return results


def compute_stats(results):
    """Calcula métricas de accuracy por bucket."""
    stats = {}
    for bucket in ["SEGURA", "CINCUENTA", "NO_NO"]:
        subset = [r for r in results if r["bucket"] == bucket]
        if not subset:
            stats[bucket] = {"n": 0, "correct": 0, "accuracy": None}
            continue
        decidable = [r for r in subset if r["correct"] is not None]
        n_correct  = sum(1 for r in decidable if r["correct"])
        stats[bucket] = {
            "n"        : len(subset),
            "decidable": len(decidable),
            "correct"  : n_correct,
            "accuracy" : round(n_correct / len(decidable), 3) if decidable else None,
        }

    # Calibración por deciles
    calibration = []
    for low in range(0, 100, 10):
        high   = low + 10
        bucket = [r for r in results if low/100 <= r["prob"] < high/100]
        if bucket:
            actual_rate = sum(r["label_actual"] for r in bucket) / len(bucket)
            calibration.append({
                "range"      : f"{low}-{high}%",
                "predicted"  : (low + 5) / 100,
                "actual"     : round(actual_rate, 3),
                "n"          : len(bucket),
            })

    return stats, calibration


def generate_html(results, stats, calibration):
    """Genera el informe HTML."""

    def fmt(n):
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000:     return f"{n/1_000:.0f}K"
        return str(int(n))

    # Tabla de resultados
    rows = ""
    for r in sorted(results, key=lambda x: x["prob"], reverse=True):
        cls     = r["bucket"].lower().replace("_", "")
        correct = r["correct"]
        icon    = "✓" if correct is True else ("✗" if correct is False else "~")
        icon_cls = "hit" if correct is True else ("miss" if correct is False else "skip")
        res_str = ""
        if r["resolution"] == 1.0:   res_str = "YES"
        elif r["resolution"] == 0.0: res_str = "NO"
        else:                        res_str = "?"
        real_lbl = "★" if r["has_real_label"] else "h"

        slug_url = f"https://polymarket.com/event/{r['slug']}" if r["slug"] else "#"
        rows += f"""
        <tr class="row-{cls}">
          <td><span class="real-badge {('real' if r['has_real_label'] else 'heur')}">{real_lbl}</span></td>
          <td><a href="{slug_url}" target="_blank" class="q-link">{r['question']}</a></td>
          <td><span class="cat">{r['category'][:12]}</span></td>
          <td><strong class="prob-{cls}">{round(r['prob']*100)}%</strong></td>
          <td><span class="bucket-tag {cls}">{r['bucket']}</span></td>
          <td>{round(r['price_yes']*100)}¢</td>
          <td class="res-{res_str.lower()}">{res_str}</td>
          <td><span class="verdict {icon_cls}">{icon}</span></td>
          <td class="dim">{r['collected_at']}</td>
        </tr>"""

    # Stats cards
    def stat_card(bucket, s, color):
        if s["n"] == 0:
            return f'<div class="stat-card"><div class="stat-bucket {color}">{bucket}</div><div class="stat-n">0 mercados</div></div>'
        acc    = s.get("accuracy")
        acc_s  = f"{acc:.0%}" if acc is not None else "N/A"
        bar    = int((acc or 0) * 100)
        return f"""
        <div class="stat-card">
          <div class="stat-bucket {color}">{bucket}</div>
          <div class="stat-acc">{acc_s}</div>
          <div class="stat-bar-wrap"><div class="stat-bar {color}" style="width:{bar}%"></div></div>
          <div class="stat-detail">{s['correct']}/{s.get('decidable',0)} correctos · {s['n']} total</div>
        </div>"""

    cards = (
        stat_card("APUESTA SEGURA", stats["SEGURA"], "green") +
        stat_card("50 / 50", stats["CINCUENTA"], "yellow") +
        stat_card("NO NO", stats["NO_NO"], "red")
    )

    # Calibración chart (SVG simple)
    cal_bars = ""
    for c in calibration:
        pred_h  = int(c["predicted"] * 100)
        act_h   = int(c["actual"] * 100)
        diff    = c["actual"] - c["predicted"]
        bar_cls = "cal-over" if diff > 0.05 else ("cal-under" if diff < -0.05 else "cal-ok")
        cal_bars += f"""
        <div class="cal-col">
          <div class="cal-actual {bar_cls}" style="height:{act_h}px" title="{c['range']}: actual={c['actual']:.0%} pred={c['predicted']:.0%} n={c['n']}"></div>
          <div class="cal-pred-line" style="bottom:{pred_h}px"></div>
          <div class="cal-label">{c['range']}</div>
          <div class="cal-n">n={c['n']}</div>
        </div>"""

    total   = len(results)
    real_n  = sum(1 for r in results if r["has_real_label"])
    hits    = sum(1 for r in results if r["correct"] is True)
    misses  = sum(1 for r in results if r["correct"] is False)
    overall = f"{hits/(hits+misses):.0%}" if (hits+misses) > 0 else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>POLYMARKET // BACKTEST REPORT</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  :root {{
    --bg:    #07090d;
    --bg2:   #0d1117;
    --bg3:   #111820;
    --line:  #1a2535;
    --text:  #b8c8d8;
    --dim:   #3a4a5a;
    --green: #00e676;
    --yellow:#ffd600;
    --red:   #ff1744;
    --blue:  #00b0ff;
    --mono:  'IBM Plex Mono', monospace;
    --sans:  'IBM Plex Sans', sans-serif;
  }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--mono); min-height: 100vh; }}

  header {{
    padding: 2rem 3rem 1.5rem;
    border-bottom: 1px solid var(--line);
    display: flex; align-items: flex-end; justify-content: space-between; flex-wrap: wrap; gap: 1rem;
  }}
  .logo {{ font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; }}
  .logo span {{ color: var(--blue); }}
  .meta {{ font-size: 0.65rem; color: var(--dim); text-align: right; line-height: 1.8; }}
  .meta strong {{ color: var(--text); }}

  main {{ padding: 2rem 3rem; max-width: 1400px; }}

  h2 {{ font-size: 0.65rem; font-weight: 600; letter-spacing: 0.2em; color: var(--dim);
        text-transform: uppercase; margin: 2rem 0 1rem; border-left: 2px solid var(--blue);
        padding-left: 0.75rem; }}

  /* ── Summary bar ── */
  .summary {{
    display: flex; gap: 0; border: 1px solid var(--line); border-radius: 4px;
    overflow: hidden; margin-bottom: 2rem;
  }}
  .sum-item {{ flex: 1; padding: 1rem 1.5rem; border-right: 1px solid var(--line); }}
  .sum-item:last-child {{ border-right: none; }}
  .sum-label {{ font-size: 0.55rem; color: var(--dim); letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.3rem; }}
  .sum-value {{ font-size: 1.6rem; font-weight: 700; }}
  .sum-value.green  {{ color: var(--green); }}
  .sum-value.yellow {{ color: var(--yellow); }}
  .sum-value.blue   {{ color: var(--blue); }}

  /* ── Stat cards ── */
  .stat-cards {{ display: flex; gap: 1.5rem; margin-bottom: 2rem; flex-wrap: wrap; }}
  .stat-card {{
    flex: 1; min-width: 200px; padding: 1.5rem;
    border: 1px solid var(--line); border-radius: 4px; background: var(--bg2);
  }}
  .stat-bucket {{ font-size: 0.55rem; letter-spacing: 0.18em; font-weight: 700; margin-bottom: 0.5rem; }}
  .stat-bucket.green  {{ color: var(--green); }}
  .stat-bucket.yellow {{ color: var(--yellow); }}
  .stat-bucket.red    {{ color: var(--red); }}
  .stat-acc {{ font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; }}
  .stat-bar-wrap {{ height: 3px; background: var(--line); border-radius: 2px; margin-bottom: 0.5rem; }}
  .stat-bar {{ height: 100%; border-radius: 2px; }}
  .stat-bar.green  {{ background: var(--green); }}
  .stat-bar.yellow {{ background: var(--yellow); }}
  .stat-bar.red    {{ background: var(--red); }}
  .stat-detail {{ font-size: 0.60rem; color: var(--dim); }}

  /* ── Calibration ── */
  .cal-wrap {{
    display: flex; align-items: flex-end; gap: 4px; height: 120px;
    padding: 0 1rem 2rem; border: 1px solid var(--line); border-radius: 4px;
    background: var(--bg2); position: relative; margin-bottom: 2rem; overflow-x: auto;
  }}
  .cal-col {{ position: relative; display: flex; flex-direction: column; align-items: center; width: 60px; flex-shrink: 0; }}
  .cal-actual {{ width: 40px; border-radius: 2px 2px 0 0; min-height: 2px; }}
  .cal-ok    {{ background: var(--green); opacity: 0.8; }}
  .cal-over  {{ background: var(--blue); opacity: 0.8; }}
  .cal-under {{ background: var(--red); opacity: 0.8; }}
  .cal-pred-line {{
    position: absolute; left: 8px; right: 8px; height: 2px;
    background: var(--yellow); opacity: 0.6; bottom: 0;
  }}
  .cal-label {{ font-size: 0.45rem; color: var(--dim); margin-top: 4px; white-space: nowrap; }}
  .cal-n     {{ font-size: 0.40rem; color: var(--dim); }}
  .cal-legend {{ font-size: 0.55rem; color: var(--dim); margin-bottom: 0.5rem; }}
  .cal-legend span {{ display: inline-block; width: 10px; height: 10px; border-radius: 1px; margin-right: 4px; vertical-align: middle; }}

  /* ── Table ── */
  .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 4px; margin-bottom: 3rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.62rem; }}
  thead th {{
    padding: 0.6rem 0.8rem; text-align: left; background: var(--bg3);
    color: var(--dim); font-size: 0.50rem; letter-spacing: 0.15em; text-transform: uppercase;
    border-bottom: 1px solid var(--line); white-space: nowrap;
  }}
  tbody tr {{ border-bottom: 1px solid var(--line); transition: background 0.1s; }}
  tbody tr:hover {{ background: var(--bg3); }}
  tbody tr:last-child {{ border-bottom: none; }}
  td {{ padding: 0.5rem 0.8rem; vertical-align: middle; }}

  .q-link {{ color: var(--text); text-decoration: none; }}
  .q-link:hover {{ color: var(--blue); }}
  .cat {{ font-size: 0.50rem; color: var(--dim); }}
  .dim {{ color: var(--dim); }}

  .prob-segura   {{ color: var(--green); }}
  .prob-cincuenta{{ color: var(--yellow); }}
  .prob-nono     {{ color: var(--red); }}

  .bucket-tag {{ font-size: 0.48rem; font-weight: 700; letter-spacing: 0.1em; padding: 2px 6px; border-radius: 2px; }}
  .bucket-tag.segura    {{ background: rgba(0,230,118,0.12); color: var(--green); }}
  .bucket-tag.cincuenta {{ background: rgba(255,214,0,0.12);  color: var(--yellow); }}
  .bucket-tag.nono      {{ background: rgba(255,23,68,0.12);  color: var(--red); }}

  .res-yes {{ color: var(--green); font-weight: 600; }}
  .res-no  {{ color: var(--red);   font-weight: 600; }}

  .verdict {{ font-size: 0.9rem; font-weight: 700; }}
  .verdict.hit  {{ color: var(--green); }}
  .verdict.miss {{ color: var(--red); }}
  .verdict.skip {{ color: var(--dim); }}

  .real-badge {{ font-size: 0.48rem; padding: 1px 4px; border-radius: 2px; font-weight: 700; }}
  .real-badge.real {{ background: rgba(0,176,255,0.15); color: var(--blue); }}
  .real-badge.heur {{ background: var(--line); color: var(--dim); }}

  /* ── Filter bar ── */
  .filters {{ display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap; }}
  .filter-btn {{
    background: none; border: 1px solid var(--line); border-radius: 2px;
    padding: 4px 12px; color: var(--dim); font-family: var(--mono); font-size: 0.55rem;
    cursor: pointer; transition: all 0.15s; letter-spacing: 0.1em;
  }}
  .filter-btn:hover {{ border-color: var(--blue); color: var(--blue); }}
  .filter-btn.active {{ border-color: var(--blue); color: var(--blue); background: rgba(0,176,255,0.08); }}
  .filter-btn.green.active  {{ border-color: var(--green); color: var(--green); background: rgba(0,230,118,0.08); }}
  .filter-btn.yellow.active {{ border-color: var(--yellow); color: var(--yellow); background: rgba(255,214,0,0.08); }}
  .filter-btn.red.active    {{ border-color: var(--red); color: var(--red); background: rgba(255,23,68,0.08); }}
</style>
</head>
<body>
<header>
  <div class="logo">POLYMARKET <span>//</span> BACKTEST</div>
  <div class="meta">
    Generado: <strong>{datetime.now().strftime('%Y-%m-%d %H:%M')}</strong><br>
    Mercados analizados: <strong>{total}</strong> · Con label real: <strong>{real_n}</strong>
  </div>
</header>
<main>

  <div class="summary">
    <div class="sum-item"><div class="sum-label">Accuracy global</div><div class="sum-value blue">{overall}</div></div>
    <div class="sum-item"><div class="sum-label">Predicciones correctas</div><div class="sum-value green">{hits}</div></div>
    <div class="sum-item"><div class="sum-label">Predicciones incorrectas</div><div class="sum-value" style="color:var(--red)">{misses}</div></div>
    <div class="sum-item"><div class="sum-label">Inconclusos (50/50)</div><div class="sum-value yellow">{total-hits-misses}</div></div>
    <div class="sum-item"><div class="sum-label">Labels reales vs heurísticos</div><div class="sum-value" style="font-size:1rem;padding-top:0.3rem">{real_n} <span style="color:var(--dim)">/ {total}</span></div></div>
  </div>

  <h2>accuracy por bucket</h2>
  <div class="stat-cards">{cards}</div>

  <h2>calibración — predicho vs real</h2>
  <div class="cal-legend">
    <span style="background:var(--green)"></span>bien calibrado &nbsp;
    <span style="background:var(--blue)"></span>sobreestima &nbsp;
    <span style="background:var(--red)"></span>subestima &nbsp;
    <span style="background:var(--yellow)"></span>predicho (línea)
  </div>
  <div class="cal-wrap">{cal_bars}</div>

  <h2>detalle de mercados resueltos</h2>
  <div class="filters">
    <button class="filter-btn active" onclick="filterTable('ALL', this)">TODOS</button>
    <button class="filter-btn green"  onclick="filterTable('SEGURA', this)">APUESTA SEGURA</button>
    <button class="filter-btn yellow" onclick="filterTable('CINCUENTA', this)">50/50</button>
    <button class="filter-btn red"    onclick="filterTable('NO_NO', this)">NO NO</button>
    <button class="filter-btn"        onclick="filterTable('HIT', this)">✓ CORRECTOS</button>
    <button class="filter-btn"        onclick="filterTable('MISS', this)">✗ INCORRECTOS</button>
    <button class="filter-btn"        onclick="filterTable('REAL', this)">★ LABEL REAL</button>
  </div>
  <div class="table-wrap">
    <table id="main-table">
      <thead>
        <tr>
          <th></th>
          <th>Mercado</th>
          <th>Categoría</th>
          <th>Prob</th>
          <th>Bucket</th>
          <th>Precio YES</th>
          <th>Resolvió</th>
          <th>¿Acertó?</th>
          <th>Fecha</th>
        </tr>
      </thead>
      <tbody id="table-body">
        {rows}
      </tbody>
    </table>
  </div>

  <div style="font-size:0.58rem;color:var(--dim);padding-bottom:3rem;line-height:1.8">
    <strong style="color:var(--text)">Notas:</strong><br>
    ★ = label real (mercado resuelto con outcome conocido) &nbsp;|&nbsp;
    h = label heurístico (proxy basado en características del mercado)<br>
    ✓ = predicción correcta &nbsp;|&nbsp; ✗ = predicción incorrecta &nbsp;|&nbsp; ~ = bucket 50/50 (inconclusivo)<br>
    SEGURA correcta = predijo bueno y era bueno &nbsp;|&nbsp; NO NO correcta = predijo malo y era malo
  </div>
</main>

<script>
function filterTable(type, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const rows = document.querySelectorAll('#table-body tr');
  rows.forEach(row => {{
    let show = true;
    if (type === 'SEGURA')   show = row.classList.contains('row-segura');
    if (type === 'CINCUENTA')show = row.classList.contains('row-cincuenta');
    if (type === 'NO_NO')    show = row.classList.contains('row-nono');
    if (type === 'HIT')      show = row.querySelector('.verdict.hit') !== null;
    if (type === 'MISS')     show = row.querySelector('.verdict.miss') !== null;
    if (type === 'REAL')     show = row.querySelector('.real-badge.real') !== null;
    row.style.display = show ? '' : 'none';
  }});
}}
</script>
</body>
</html>"""

    return html


if __name__ == "__main__":
    log.info("Ejecutando backtest...")
    results = run_backtest()

    if not results:
        print("Sin datos para el backtest. Ejecuta main.py primero para acumular datos.")
    else:
        stats, calibration = compute_stats(results)
        html = generate_html(results, stats, calibration)

        out = Path("backtest_report.html")
        out.write_text(html, encoding="utf-8")
        log.info(f"Informe guardado: {out.absolute()}")
        print(f"\n✅ Informe generado: {out.absolute()}")
        print(f"   Mercados analizados: {len(results)}")

        for bucket, s in stats.items():
            acc = f"{s['accuracy']:.0%}" if s.get("accuracy") is not None else "N/A"
            print(f"   {bucket:12s}: {acc} ({s.get('correct',0)}/{s.get('decidable',0)} correctos, {s['n']} total)")

        print(f"\n   Abre backtest_report.html en tu navegador.")