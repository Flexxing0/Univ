from lab3_punto1 import run_parte_a, run_parte_b, plot_comparison
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors

#Constantes del entorno 
LAKE_MAP   = ["SFFF", "FHFH", "FFFH", "HFFG"]
CELL_LABEL = {s: LAKE_MAP[s // 4][s % 4] for s in range(16)}
ARROW      = {0: "←", 1: "↓", 2: "→", 3: "↑"}
ACTION_HDR = ["← (0)", "↓ (1)", "→ (2)", "↑ (3)"]

#Obtener resultados
vi_results = run_parte_a()
ql_results = run_parte_b(vi_results)
plot_comparison(vi_results, ql_results)

V_det      = vi_results["Deterministico"]["V"]
pol_vi_det = vi_results["Deterministico"]["policy"]
V_sto      = vi_results["Estocastico"]["V"]
pol_vi_sto = vi_results["Estocastico"]["policy"]

Q_det      = ql_results["Deterministico"]["Q"]
pol_ql_det = ql_results["Deterministico"]["policy"]
Q_sto      = ql_results["Estocastico"]["Q"]
pol_ql_sto = ql_results["Estocastico"]["policy"]

# Funciones build

def build_value_table(V):
    rows = [["Estado", "Celda", "V*(s)"]]      # header
    for s in range(len(V)):
        rows.append([
            str(s),
            CELL_LABEL[s],
            f"{V[s]:.4f}"
        ])
    return rows

def build_policy_table(policy):
    rows = [["Estado", "Celda", "Acción", "Dirección"]]    # header
    for s in range(len(policy)):
        a = int(policy[s])
        rows.append([
            str(s),
            CELL_LABEL[s],
            str(a),
            ARROW[a]
        ])
    return rows

def build_q_table(Q):
    n_states, n_actions = Q.shape
    rows = [["Estado", "Celda"] + ACTION_HDR + ["Mejor acción"]]    # header
    highlight_cols = [None]     # None para el header
    for s in range(n_states):
        best_a = int(np.argmax(Q[s]))
        rows.append([
            str(s),
            CELL_LABEL[s],
            f"{Q[s,0]:.4f}",
            f"{Q[s,1]:.4f}",
            f"{Q[s,2]:.4f}",
            f"{Q[s,3]:.4f}",
            f"{ARROW[best_a]} ({best_a})"
        ])
        highlight_cols.append(best_a + 2)  
    return rows, highlight_cols

# ── Estilo base para todas las tablas ──────────────────────────

def base_style():
    return TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#1565c0")),  # header azul
        ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#e3f2fd"), colors.white]),
        ("FONTSIZE",      (0,1), (-1,-1), 9),
        ("GRID",          (0,0), (-1,-1), 0.4, colors.HexColor("#90caf9")),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ])

# ── Funciones make_reportlab_* ─────────────────────────────────

def make_reportlab_value_table(V):
    rows = build_value_table(V)
    t = Table(rows, colWidths=[2.5*cm, 2*cm, 2.5*cm])
    ts = base_style()
    # colorear la columna Celda según tipo
    CELL_BG = {"H": "#ffcdd2", "G": "#c8e6c9", "S": "#b3e5fc"}
    for row_idx in range(1, len(rows)):
        s = row_idx - 1
        bg = CELL_BG.get(CELL_LABEL[s])
        if bg:
            ts.add("BACKGROUND", (1, row_idx), (1, row_idx), colors.HexColor(bg))
    t.setStyle(ts)
    return t

def make_reportlab_policy_table(policy):
    rows = build_policy_table(policy)
    t = Table(rows, colWidths=[2.5*cm, 2*cm, 2.5*cm, 2.5*cm])
    ts = base_style()
    CELL_BG = {"H": "#ffcdd2", "G": "#c8e6c9", "S": "#b3e5fc"}
    for row_idx in range(1, len(rows)):
        s = row_idx - 1
        bg = CELL_BG.get(CELL_LABEL[s])
        if bg:
            ts.add("BACKGROUND", (1, row_idx), (1, row_idx), colors.HexColor(bg))
    t.setStyle(ts)
    return t

def make_reportlab_q_table(Q, highlight_color="#bbdefb"):
    rows, highlight_cols = build_q_table(Q)
    t = Table(rows, colWidths=[1.8*cm, 1.5*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.2*cm, 2.8*cm])
    ts = base_style()
    CELL_BG = {"H": "#ffcdd2", "G": "#c8e6c9", "S": "#b3e5fc"}
    for row_idx in range(1, len(rows)):
        s = row_idx - 1
        # color de celda
        bg = CELL_BG.get(CELL_LABEL[s])
        if bg:
            ts.add("BACKGROUND", (1, row_idx), (1, row_idx), colors.HexColor(bg))
        # resaltar columna del máximo
        col = highlight_cols[row_idx]
        ts.add("BACKGROUND", (col, row_idx), (col, row_idx), colors.HexColor(highlight_color))
        ts.add("FONTNAME",   (col, row_idx), (col, row_idx), "Helvetica-Bold")
    t.setStyle(ts)
    return t

# Arma el PDF

styles = getSampleStyleSheet()
h1      = styles["Heading1"]
h2      = styles["Heading2"]
normal  = styles["Normal"]
caption = ParagraphStyle("caption", parent=normal, fontSize=9,
                          textColor=colors.HexColor("#546e7a"), alignment=1)

story = []

# Título
story.append(Paragraph("Resultados – Punto 1: Frozen Lake", styles["Title"]))
story.append(Spacer(1, 0.5*cm))

# Sección 1: Value Iteration
story.append(Paragraph("1. Tabla de Valores – Value Iteration", h1))

story.append(Paragraph("Modo Determinístico", h2))
story.append(make_reportlab_value_table(V_det))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("Modo Estocástico", h2))
story.append(make_reportlab_value_table(V_sto))
story.append(Spacer(1, 0.3*cm))

# Sección 2: Políticas VI
story.append(Paragraph("2. Política Óptima – Value Iteration", h1))

story.append(Paragraph("Modo Determinístico", h2))
story.append(make_reportlab_policy_table(pol_vi_det))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("Modo Estocástico", h2))
story.append(make_reportlab_policy_table(pol_vi_sto))

# Sección 3: Tabla Q
story.append(PageBreak())
story.append(Paragraph("3. Tabla Q Final – Q-Learning", h1))

story.append(Paragraph("Modo Determinístico", h2))
story.append(make_reportlab_q_table(Q_det, highlight_color="#bbdefb"))
story.append(Spacer(1, 0.3*cm))

story.append(Paragraph("Modo Estocástico", h2))
story.append(make_reportlab_q_table(Q_sto, highlight_color="#ffccbc"))

# Generar el PDF
doc = SimpleDocTemplate("resultados_punto1.pdf")
doc.build(story)
print("PDF generado: resultados_punto1.pdf")