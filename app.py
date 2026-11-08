import base64
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as pgo
import streamlit as st
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.datasets import load_digits, load_iris, make_circles, make_moons
from sklearn.model_selection import train_test_split

from metrics import classification_report, confusion_matrix
from network import predict, train
from preprocessing import Preprocessor

st.set_page_config(page_title="Neural network from scratch", page_icon="🧠", layout="wide")

# Background: a faint neural-network drawing. Cards: one blue family, so everything matches.
def network_background():
    """SVG of a small fully connected network, used as the page background."""
    layers = [4, 6, 6, 3]
    w, h = 1600, 900
    xs = [w * (0.12 + 0.76 * i / (len(layers) - 1)) for i in range(len(layers))]
    pts = [[(xs[i], h * (0.5 + (j - (n - 1) / 2) * 0.12)) for j in range(n)] for i, n in enumerate(layers)]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid slice">']
    for i in range(len(layers) - 1):
        for (x1, y1) in pts[i]:
            for (x2, y2) in pts[i + 1]:
                parts.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="#4F7BFF" stroke-opacity="0.14" stroke-width="1.4"/>')
    for layer in pts:
        for (x, y) in layer:
            parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="15" fill="#4F7BFF" fill-opacity="0.10" stroke="#4F7BFF" stroke-opacity="0.28" stroke-width="2"/>')
    parts.append("</svg>")
    return base64.b64encode("".join(parts).encode()).decode()


st.markdown(f"""
<style>
/* Faint network drawing behind the header only. It fades out so it never sits behind results. */
[data-testid="stMainBlockContainer"] {{ position: relative; isolation: isolate; }}
[data-testid="stMainBlockContainer"]::before {{
    content: ""; position: absolute; z-index: -1; pointer-events: none;
    top: -6rem; left: -6rem; right: -6rem; height: 640px;
    background: url("data:image/svg+xml;base64,{network_background()}") center top / cover no-repeat;
    -webkit-mask-image: linear-gradient(to bottom, #000 30%, transparent 100%);
    mask-image: linear-gradient(to bottom, #000 30%, transparent 100%);
}}
.hero h1 {{ font-size: 2.7rem; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 0.3rem 0; padding: 0; }}
.hero h1 span {{ background: linear-gradient(90deg, #2B59E0, #4F46E5 60%, #0E82C8); -webkit-background-clip: text; background-clip: text; color: transparent; }}
.hero p {{ font-size: 1.05rem; opacity: 0.8; margin: 0 0 0.9rem 0; max-width: 46rem; }}
.pills span {{ display: inline-block; margin: 0 0.4rem 0.4rem 0; padding: 0.22rem 0.75rem; border-radius: 999px; font-size: 0.82rem; font-weight: 600;
    color: #2B59E0; background: rgba(43,89,224,0.10); border: 1px solid rgba(43,89,224,0.28); }}
[class*="st-key-card_"] {{ border-radius: 14px; backdrop-filter: blur(6px); box-shadow: 0 2px 12px rgba(30,60,160,0.10); }}
.st-key-card_data  {{ background: linear-gradient(145deg, rgba(43,89,224,0.16), rgba(43,89,224,0.04)); border-color: rgba(43,89,224,0.35); }}
.st-key-card_arch  {{ background: linear-gradient(145deg, rgba(79,70,229,0.16), rgba(79,70,229,0.04)); border-color: rgba(79,70,229,0.35); }}
.st-key-card_train {{ background: linear-gradient(145deg, rgba(14,130,200,0.16), rgba(14,130,200,0.04)); border-color: rgba(14,130,200,0.35); }}
.st-key-card_shape {{ background: linear-gradient(120deg, rgba(43,89,224,0.12), rgba(14,130,200,0.06)); border-color: rgba(43,89,224,0.30); }}
.flow {{ display: flex; align-items: center; flex-wrap: wrap; gap: 0.5rem; margin: 0.6rem 0 0.4rem 0; }}
.lyr {{ width: 120px; height: 86px; box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 0.4rem 0.5rem; border-radius: 12px; color: #fff; line-height: 1.2; box-shadow: 0 2px 8px rgba(30,60,160,0.18); }}
.lyr b {{ display: block; font-size: 1.4rem; }}
.lyr span {{ display: block; font-size: 0.78rem; opacity: 0.92; }}
.lyr small {{ display: block; font-size: 0.72rem; opacity: 0.85; font-family: monospace; }}
.lyr.in {{ background: linear-gradient(135deg, #64748B, #475569); }}
.lyr.hid {{ background: linear-gradient(135deg, #4F46E5, #2B59E0); }}
.lyr.out {{ background: linear-gradient(135deg, #0D9488, #0E82C8); }}
.arrow {{ opacity: 0.5; font-size: 1.2rem; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 0.3rem 1.2rem; font-size: 0.85rem; opacity: 0.85; margin: 0.2rem 0 0.4rem 0; }}
.legend .dot {{ display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 50%; margin-right: 0.35rem; }}
.empty {{ border: 2px dashed rgba(43,89,224,0.35); border-radius: 14px; padding: 1.5rem 1rem; text-align: center; background: rgba(43,89,224,0.04); }}
.empty > b {{ font-size: 1.15rem; }}
.steps {{ margin-top: 0.8rem; display: flex; justify-content: center; gap: 0.6rem; flex-wrap: wrap; }}
.steps span {{ padding: 0.28rem 0.9rem; border-radius: 999px; background: rgba(43,89,224,0.10); font-size: 0.88rem; }}
.st-key-train button, .st-key-predict button {{ background: linear-gradient(135deg, #2B59E0, #4F46E5); border: 0; color: #fff; font-weight: 600; }}
.st-key-train button:hover:not(:disabled), .st-key-predict button:hover:not(:disabled) {{ filter: brightness(1.12); color: #fff; }}
.st-key-results [data-testid="stMetric"] {{ background: linear-gradient(145deg, rgba(43,89,224,0.08), rgba(43,89,224,0.02)); border-radius: 12px; }}
</style>
""", unsafe_allow_html=True)

MAX_CATEGORIES = 20   # a text column with more distinct values than this is treated as an ID / free text
HIDDEN_ACTIVATIONS = ["sigmoid", "tanh", "relu", "leaky_relu", "linear"]
DATASETS = ["XOR", "Moons", "Circles", "Iris", "Digits", "Upload CSV"]
DATASET_NOTES = {
    "XOR": "Two noisy inputs, class 1 when exactly one of them is high. Needs a hidden layer.",
    "Moons": "Two interleaving half circles. A non-linear boundary is required.",
    "Circles": "A ring around a smaller ring. Another non-linear case.",
    "Iris": "150 flowers, 4 measurements, 3 species.",
    "Digits": "1,797 handwritten digits as 8x8 pixel images, 10 classes.",
}
BLUE, AMBER, RED = "#2B59E0", "#E08A1E", "#E0455B"
PALETTE = ["#E0455B", "#2B59E0", "#16A34A", "#F59E0B", "#8B5CF6", "#92400E", "#0D9488", "#6B7280", "#84CC16", "#111827"]


def pale(hex_colour, amount=0.75):
    """Mix a colour with white, used for soft decision-boundary regions."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    mix = lambda c: int(c + (255 - c) * amount)
    return f"rgb({mix(r)},{mix(g)},{mix(b)})"



@st.cache_data
def read_csv(raw):
    return pd.read_csv(io.BytesIO(raw))


def split_data(X_df, y, test_size):
    """Train/test split. Keeps the class proportions equal in both parts when that is possible."""
    try:
        return train_test_split(X_df, y, test_size=test_size, random_state=42, stratify=y)
    except ValueError:      # a class has too few samples to stratify
        return train_test_split(X_df, y, test_size=test_size, random_state=42)


class DataError(Exception):
    """Raised when an uploaded CSV cannot be used. The message is shown to the user."""


@st.cache_data
def load_data(name):
    if name == "XOR":
        rng = np.random.default_rng(0)
        X = rng.integers(0, 2, size=(400, 2)).astype(float)
        y = np.logical_xor(X[:, 0], X[:, 1]).astype(int)
        return X + rng.normal(0, 0.1, X.shape), y
    if name == "Moons":
        return make_moons(500, noise=0.15, random_state=0)
    if name == "Circles":
        return make_circles(500, noise=0.08, factor=0.5, random_state=0)
    if name == "Iris":
        d = load_iris()
        return d.data, d.target
    d = load_digits()
    return d.data, d.target


def prepare_csv(df, target):
    """Clean an uploaded CSV. Returns (features DataFrame, target codes, class names).

    Missing feature values are NOT dropped here: the Preprocessor fills them later, using
    statistics of the training split only."""
    n_rows = len(df)
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if len(df) < n_rows:
        st.warning(f"Dropped {n_rows - len(df)} row(s) where the target column is empty.")

    keep, ignored = [], []
    for c in df.columns.drop(target):
        col = df[c]
        if col.isna().all():
            ignored.append(f"{c} (empty)")
        elif is_numeric_dtype(col) and not is_bool_dtype(col):
            keep.append(c)
        elif col.nunique() <= MAX_CATEGORIES:
            keep.append(c)                       # text / boolean column with few values -> categorical
        else:
            ignored.append(f"{c} (text with {col.nunique()} different values, looks like an ID or free text)")
    if ignored:
        st.warning("Ignoring columns: " + "; ".join(ignored))
    if not keep:
        raise DataError("The CSV has no usable feature columns.")
    if len(df) < 20:
        raise DataError("Need at least 20 rows with a target value.")

    feats = df[keep]
    n_missing = int(feats.isna().sum().sum())
    if n_missing:
        n_cols = int(feats.isna().any().sum())
        st.info(f"{n_missing} missing value(s) in {n_cols} column(s) will be filled using training-data "
                "statistics only (numeric: median, categorical: most frequent value).")

    codes, classes = pd.factorize(df[target])
    if len(classes) < 2:
        raise DataError("The target column needs at least 2 different values.")
    if len(classes) > 20:
        raise DataError(f"The target column has {len(classes)} unique values, which looks like a number to "
                        "predict (regression). Only classification is supported right now.")
    return feats, codes, [str(c) for c in classes]


# ---------------- charts ----------------
def style_fig(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=55, r=20, t=55, b=55), plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(automargin=True, gridcolor="rgba(128,128,128,0.18)", linecolor="rgba(128,128,128,0.35)")
    fig.update_yaxes(automargin=True, gridcolor="rgba(128,128,128,0.18)", linecolor="rgba(128,128,128,0.35)")
    return fig


def curve_fig(train_values, test_values, title):
    epochs = list(range(1, len(train_values) + 1))
    fig = pgo.Figure()
    fig.add_trace(pgo.Scatter(x=epochs, y=train_values, name="Train", mode="lines", line=dict(color=BLUE, width=2.5)))
    fig.add_trace(pgo.Scatter(x=epochs, y=test_values, name="Test", mode="lines", line=dict(color=AMBER, width=2.5)))
    style_fig(fig, 340)
    fig.update_layout(title=dict(text=title, font=dict(size=15), x=0, y=0.97), hovermode="x unified")
    fig.update_xaxes(title="Epoch", zeroline=False)
    fig.update_yaxes(zeroline=False)
    return fig


def draw_network(layer_sizes, activations, W, max_show=10):
    """Neurons as circles, weights as lines: blue = positive, red = negative, thicker and darker = bigger."""
    shown = [min(n, max_show) for n in layer_sizes]
    n_layers = len(layer_sizes)
    pos = []
    for l, n in enumerate(shown):
        ys = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) if n > 1 else np.array([0.0])
        pos.append((np.full(n, float(l)), ys))

    widths, opacities = (0.6, 1.4, 2.8), (0.18, 0.45, 0.9)      # weak, medium, strong weights
    segs = {(s, k): ([], []) for s in (1, -1) for k in range(3)}
    for l in range(n_layers - 1):
        w = W[l][: shown[l], : shown[l + 1]]
        wmax = np.abs(w).max() + 1e-12
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                s = 1 if w[i, j] >= 0 else -1
                k = min(int(abs(w[i, j]) / wmax * 3), 2)
                xs, ys = segs[(s, k)]
                xs += [pos[l][0][i], pos[l + 1][0][j], None]
                ys += [pos[l][1][i], pos[l + 1][1][j], None]

    fig = pgo.Figure()
    for (s, k), (xs, ys) in segs.items():
        fig.add_trace(pgo.Scatter(
            x=xs, y=ys, mode="lines", opacity=opacities[k], hoverinfo="skip", showlegend=False,
            line=dict(color=BLUE if s == 1 else RED, width=widths[k]),
        ))
    notes = []
    top = max(p[1].max() for p in pos) + 1.0          # one common row for all layer labels
    size = 18 if max(shown) <= 8 else 14 if max(shown) <= 20 else 10
    for l in range(n_layers):
        colour = "#64748B" if l == 0 else "#0D9488" if l == n_layers - 1 else "#4F46E5"
        fig.add_trace(pgo.Scatter(
            x=pos[l][0], y=pos[l][1], mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(size=size, color=colour, line=dict(color="white", width=1.5)),
        ))
        role = "Input" if l == 0 else "Output" if l == n_layers - 1 else f"Hidden {l}"
        detail = f"{layer_sizes[l]} {'neuron' if layer_sizes[l] == 1 else 'neurons'}"
        if l:
            detail += f" · {activations[l - 1]}"
        extra = f"<br><i>first {max_show} shown</i>" if layer_sizes[l] > max_show else ""
        notes.append(dict(x=l, y=top, text=f"<b>{role}</b><br>{detail}{extra}", showarrow=False, font=dict(size=13), yanchor="bottom"))
    ymin = min(p[1].min() for p in pos)
    style_fig(fig, 480)
    fig.update_layout(annotations=notes, plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False, range=[-0.4, n_layers - 0.6]),
                      yaxis=dict(visible=False, range=[ymin - 0.8, top + 1.9]))
    return fig


def layer_flow_html(layer_sizes, activations):
    """One block per layer, joined by arrows. Colours match the network diagram."""
    n, parts = len(layer_sizes), []
    for i, size in enumerate(layer_sizes):
        kind = "in" if i == 0 else "out" if i == n - 1 else "hid"
        role = "Input" if i == 0 else "Output" if i == n - 1 else f"Hidden {i}"
        act = "" if i == 0 else f"<small>{activations[i - 1]}</small>"
        if i:
            parts.append('<span class="arrow">→</span>')
        parts.append(f'<div class="lyr {kind}"><b>{size}</b><span>{role}</span>{act}</div>')
    return '<div class="flow">' + "".join(parts) + "</div>"


def weights_fig(Wm, row_labels):
    """Heatmap of one weight matrix: blue = positive, red = negative, white = near zero."""
    lim = float(np.abs(Wm).max()) or 1.0
    fig = pgo.Figure(pgo.Heatmap(
        z=Wm, x=[f"n{j + 1}" for j in range(Wm.shape[1])], y=[str(r) for r in row_labels], zmin=-lim, zmax=lim,
        colorscale=[[0, RED], [0.5, "rgb(255,255,255)"], [1, BLUE]], colorbar=dict(thickness=12, title="weight"),
        hovertemplate="from %{y} to %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(height=int(min(620, 110 + 22 * Wm.shape[0])), margin=dict(l=110, r=10, t=30, b=10),
                      xaxis=dict(type="category", side="top"),
                      yaxis=dict(type="category", autorange="reversed", automargin=True))
    return fig


def confusion_fig(cm, class_names):
    """Heatmap of the confusion matrix. Rows = true class, columns = predicted class."""
    fig = pgo.Figure(pgo.Heatmap(
        z=cm, x=class_names, y=class_names, colorscale="Blues", showscale=False,
        text=cm, texttemplate="%{text}", textfont=dict(size=15), hovertemplate="True %{y}, predicted %{x}: %{z}<extra></extra>",
    ))
    fig.update_layout(height=max(320, 70 * len(class_names) + 120), margin=dict(l=60, r=20, t=30, b=60),
                      xaxis=dict(title="Predicted class", type="category", side="bottom"),
                      yaxis=dict(title="True class", type="category", autorange="reversed"))
    return fig


def draw_boundary(run):
    """Soft colour map of what the trained network predicts at every point of the 2D plane."""
    Xtr_df, Xte_df, ytr, yte = run["Xtr_df"], run["Xte_df"], run["ytr"], run["yte"]
    c0, c1 = Xtr_df.columns[0], Xtr_df.columns[1]
    Xall = pd.concat([Xtr_df, Xte_df])
    pad0, pad1 = 0.05 * (Xall[c0].max() - Xall[c0].min()), 0.05 * (Xall[c1].max() - Xall[c1].min())
    gx = np.linspace(Xall[c0].min() - pad0, Xall[c0].max() + pad0, 150)
    gy = np.linspace(Xall[c1].min() - pad1, Xall[c1].max() + pad1, 150)
    xx, yy = np.meshgrid(gx, gy)
    grid = run["prep"].transform(pd.DataFrame({c0: xx.ravel(), c1: yy.ravel()}))
    out = predict(grid, run["W"], run["b"], run["activations"])
    n = run["n_classes"]
    binary = n == 2
    z = (out[:, 0] if binary else out.argmax(axis=1)).reshape(xx.shape)

    fig = pgo.Figure()
    if binary:
        scale = [[0, pale(PALETTE[0])], [0.5, "rgb(255,255,255)"], [1, pale(PALETTE[1])]]
        fig.add_trace(pgo.Contour(x=gx, y=gy, z=z, zmin=0, zmax=1, colorscale=scale, line_width=0, hoverinfo="skip",
                                  colorbar=dict(title=f"P({run['class_names'][1]})", thickness=12, len=0.8)))
    else:
        scale = []
        for i in range(n):
            col = pale(PALETTE[i % len(PALETTE)])
            scale += [[i / n, col], [(i + 1) / n, col]]
        fig.add_trace(pgo.Heatmap(x=gx, y=gy, z=z, zmin=-0.5, zmax=n - 0.5, colorscale=scale, showscale=False, hoverinfo="skip"))
    for c in range(n):
        col = PALETTE[c % len(PALETTE)]
        m = ytr == c
        fig.add_trace(pgo.Scatter(x=Xtr_df[c0][m], y=Xtr_df[c1][m], mode="markers", name=f"{run['class_names'][c]} (train)",
                                  marker=dict(size=6, color=col, opacity=0.55)))
        m = yte == c
        fig.add_trace(pgo.Scatter(x=Xte_df[c0][m], y=Xte_df[c1][m], mode="markers", name=f"{run['class_names'][c]} (test)",
                                  marker=dict(size=9, color=col, symbol="diamond", line=dict(color="white", width=1.5))))
    style_fig(fig, 540)
    fig.update_layout(xaxis_title=str(c0), yaxis_title=str(c1), legend=dict(orientation="h", y=1.1, x=0, xanchor="left"))
    fig.update_xaxes(showgrid=False, zeroline=False, range=[gx[0], gx[-1]])
    fig.update_yaxes(showgrid=False, zeroline=False, range=[gy[0], gy[-1]])
    return fig


# ---------------- header ----------------
st.markdown("""
<div class="hero">
<h1>Neural network <span>from scratch</span></h1>
<p>Build, train and inspect a neural network written in plain NumPy. The forward pass, backpropagation and the Adam optimiser are all hand-coded.</p>
<div class="pills"><span>NumPy only</span><span>Backpropagation</span><span>Adam optimiser</span><span>Activation per layer</span><span>CSV upload</span></div>
</div>
""", unsafe_allow_html=True)

# ---------------- configuration ----------------
col_data, col_arch, col_train = st.columns([1.05, 1.3, 1], gap="medium")
notes_area = st.container()     # warnings and errors about the data appear here, under the three cards

data_ok = False
with col_data:
    with st.container(border=True, key="card_data", height="stretch"):
        st.subheader("Data")
        dataset = st.selectbox("Dataset", DATASETS, key="dataset")
        csv_df, target_col = None, None
        if dataset == "Upload CSV":
            csv_file = st.file_uploader("CSV file", type="csv", key="csv_file")
            if csv_file is not None:
                csv_df = read_csv(csv_file.getvalue())
                target_col = st.selectbox("Column to predict", csv_df.columns, index=len(csv_df.columns) - 1, key="target_col")
        else:
            st.caption(DATASET_NOTES[dataset])
        test_size = st.slider("Test split", 0.1, 0.5, 0.2, 0.05, key="test_size")
        split_info = st.empty()

        if dataset == "Upload CSV":
            if csv_df is None:
                st.info("Upload a CSV file, then choose the column to predict.")
            else:
                try:
                    with notes_area:
                        X_df, y, class_names = prepare_csv(csv_df, target_col)
                    data_ok = True
                except DataError as e:
                    notes_area.error(str(e))
        else:
            X, y = load_data(dataset)
            X_df = pd.DataFrame(X, columns=[f"feature {i + 1}" for i in range(X.shape[1])])
            class_names = [str(c) for c in np.unique(y)]
            data_ok = True

if data_ok:
    n_classes = len(np.unique(y))
    # Split first, then fit the preprocessor on the training part only
    Xtr_df, Xte_df, ytr, yte = split_data(X_df, y, test_size)
    prep = Preprocessor().fit(Xtr_df)
    split_info.caption(f"{len(Xtr_df):,} training samples and {len(Xte_df):,} test samples")
    n_features = prep.n_features_out    # input neurons = columns after encoding
    if n_classes == 2:
        n_out, out_act, loss_name = 1, "sigmoid", "bce"
    else:
        n_out, out_act, loss_name = n_classes, "softmax", "cce"

with col_arch:
    with st.container(border=True, key="card_arch", height="stretch"):
        st.subheader("Architecture")
        n_hidden = int(st.number_input("Hidden layers", 0, 8, 2, key="n_hidden"))
        sizes, acts = [], []
        for i in range(n_hidden):
            row = st.columns([1, 1.3])
            sizes.append(row[0].number_input(f"Hidden {i + 1} neurons", 1, 512, 16, key=f"n{i}"))
            acts.append(row[1].selectbox(f"Hidden {i + 1} activation", HIDDEN_ACTIVATIONS, key=f"a{i}"))
        if data_ok:
            st.caption(f"Output layer: {n_out} neuron{'s' if n_out > 1 else ''} with {out_act}. "
                       f"The loss ({loss_name}) is chosen from the task.")

with col_train:
    with st.container(border=True, key="card_train", height="stretch"):
        st.subheader("Training")
        t1, t2 = st.columns(2)
        epochs = t1.number_input("Epochs", 1, 5000, 100, key="epochs")
        batch_size = t2.number_input("Batch size", 1, 4096, 32, key="batch_size")
        lr = st.number_input("Learning rate (Adam)", 0.00001, 1.0, 0.01, step=0.001, format="%.5f", key="lr")
        fresh_seed = st.checkbox("New random initialisation on every run", value=True, key="fresh_seed")
        seed = st.number_input("Seed", 0, 1_000_000, 42, disabled=fresh_seed, key="seed",
                               help="Used when random initialisation is switched off, so a run can be repeated exactly.")
        go = st.button("Train model", type="primary", width="stretch", disabled=not data_ok, key="train")

if not data_ok:
    st.stop()

layer_sizes = [n_features, *map(int, sizes), n_out]
activations = [*acts, out_act]
n_params = sum((layer_sizes[i] + 1) * layer_sizes[i + 1] for i in range(len(layer_sizes) - 1))

# ---------------- network shape ----------------
with st.container(border=True, key="card_shape"):
    st.subheader("Network shape")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Samples", f"{len(X_df):,}")
    f2.metric("Input neurons", n_features)
    f3.metric("Classes", n_classes)
    f4.metric("Parameters", f"{n_params:,}")
    st.markdown(layer_flow_html(layer_sizes, activations), unsafe_allow_html=True)
    st.caption(f"Loss function: {loss_name}, chosen automatically from the task.")
    if dataset == "Upload CSV":
        with st.expander("Preprocessing"):
            st.write(f"**Numeric columns ({len(prep.num_cols)}):** missing values → median, then standardised. "
                     + (", ".join(map(str, prep.num_cols)) if prep.num_cols else "none"))
            st.write(f"**Categorical columns ({len(prep.cat_cols)}):** missing values → most frequent value, then one-hot encoded. "
                     + (", ".join(f"{c} ({len(prep.categories[c])} values)" for c in prep.cat_cols) if prep.cat_cols else "none"))
            st.caption("All statistics come from the training split only.")

# ---------------- train ----------------
if go:
    Xtr, Xte = prep.transform(Xtr_df), prep.transform(Xte_df)

    if n_classes == 2:
        Ytr, Yte = ytr.reshape(-1, 1).astype(float), yte.reshape(-1, 1).astype(float)
    else:
        Ytr, Yte = np.eye(n_classes)[ytr], np.eye(n_classes)[yte]

    run_seed = int(np.random.randint(0, 1_000_000)) if fresh_seed else int(seed)

    with st.spinner("Training model..."):
        W, b, h = train(
            Xtr, Ytr, layer_sizes, activations, loss_name,
            epochs=int(epochs), batch_size=int(batch_size), lr=float(lr),
            seed=run_seed, X_val=Xte, y_val=Yte,
        )

    # Save results in session_state so they don't disappear when Predict triggers a rerun
    st.session_state["run"] = dict(
        W=W, b=b, h=h, prep=prep, activations=activations, dataset=dataset,
        n_classes=n_classes, Xtr_df=Xtr_df, Xte_df=Xte_df, ytr=ytr, yte=yte,
        layer_sizes=layer_sizes, seed=run_seed, class_names=class_names,
        signature=(dataset, tuple(map(str, X_df.columns)), len(X_df)),
    )
    st.toast("Training finished")

# ---------------- results ----------------
st.subheader("Results")
run = st.session_state.get("run")
if run is None:
    st.markdown('<div class="empty"><b>No model trained yet</b><div class="steps">'
                '<span>1 · Choose a dataset</span><span>2 · Set layers and activations</span>'
                '<span>3 · Press <b>Train model</b></span></div></div>', unsafe_allow_html=True)
    st.stop()

current = (dataset, tuple(map(str, X_df.columns)), len(X_df))
if run["signature"] != current or run["layer_sizes"] != layer_sizes or run["activations"] != activations:
    st.info(f"These results come from an earlier run ({run['dataset']}, "
            f"{' → '.join(map(str, run['layer_sizes']))}). Select Train model to run the current settings.")
st.caption(f"{run['dataset']}  |  seed {run['seed']}  |  {' → '.join(map(str, run['layer_sizes']))}  |  {', '.join(run['activations'])}")

h = run["h"]
with st.container(key="results"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test accuracy", f"{h['val_score'][-1] * 100:.1f}%", border=True)
    m2.metric("Train accuracy", f"{h['score'][-1] * 100:.1f}%", border=True)
    m3.metric("Test loss", f"{h['val_loss'][-1]:.4f}", border=True)
    m4.metric("Train loss", f"{h['loss'][-1]:.4f}", border=True)

tab_curves, tab_metrics, tab_net, tab_boundary, tab_weights, tab_try = st.tabs(
    ["Learning curves", "Metrics", "Network", "Decision boundary", "Weights", "Try it"]
)

with tab_curves:
    ca, cb = st.columns(2)
    ca.plotly_chart(curve_fig(h["loss"], h["val_loss"], "Loss"), width="stretch", theme=None, config={"displayModeBar": False})
    cb.plotly_chart(curve_fig(h["score"], h["val_score"], "Accuracy"), width="stretch", theme=None, config={"displayModeBar": False})

with tab_metrics:
    out = predict(run["prep"].transform(run["Xte_df"]), run["W"], run["b"], run["activations"])
    y_hat = (out[:, 0] > 0.5).astype(int) if run["n_classes"] == 2 else out.argmax(axis=1)
    cm = confusion_matrix(run["yte"], y_hat, run["n_classes"])
    rep = classification_report(cm)
    left, right = st.columns([1, 1])
    with left:
        st.caption("Confusion matrix on the test set. Rows are the true class, columns are what the network predicted.")
        st.plotly_chart(confusion_fig(cm, run["class_names"]), width="stretch", config={"displayModeBar": False})
    with right:
        st.caption("Per-class scores on the test set.")
        table = pd.DataFrame(
            {"Precision": rep["precision"], "Recall": rep["recall"], "F1": rep["f1"], "Samples": rep["support"]},
            index=run["class_names"],
        )
        table.loc["Macro average"] = [rep["macro"]["precision"], rep["macro"]["recall"], rep["macro"]["f1"], rep["support"].sum()]
        st.dataframe(table.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "Samples": "{:.0f}"}), width="stretch")
        st.caption("Precision: of the samples predicted as a class, how many were right. "
                   "Recall: of the samples that truly belong to a class, how many were found.")

with tab_net:
    biggest = max(run["layer_sizes"])
    if biggest <= 10:
        max_show = biggest
    else:
        max_show = st.slider("Max neurons shown per layer", 5, min(biggest, 128), min(biggest, 30), key="max_show")
    st.markdown(
        '<div class="legend">'
        '<span><span class="dot" style="background:#2B59E0"></span>Positive weight</span>'
        '<span><span class="dot" style="background:#E0455B"></span>Negative weight</span>'
        '<span><span class="dot" style="background:#64748B"></span>Input neuron</span>'
        '<span><span class="dot" style="background:#4F46E5"></span>Hidden neuron</span>'
        '<span><span class="dot" style="background:#0D9488"></span>Output neuron</span>'
        '<span>Thicker and darker lines mean larger weights</span></div>', unsafe_allow_html=True)
    st.plotly_chart(
        draw_network(run["layer_sizes"], run["activations"], run["W"], max_show=max_show),
        width="stretch", theme=None, config={"displayModeBar": False},
    )

with tab_boundary:
    if len(run["prep"].columns) == 2 and not run["prep"].cat_cols:
        st.caption("The background shows the network's prediction at each point. Dots are training samples, diamonds are test samples.")
        st.plotly_chart(draw_boundary(run), width="stretch", theme=None, config={"displayModeBar": False})
    else:
        st.info("Decision boundaries are only available for datasets with exactly 2 numeric features (XOR, Moons, Circles).")

with tab_weights:
    st.caption("Each layer's weight matrix as a heatmap: rows are the inputs to the layer, columns are its neurons. Blue is positive, red is negative, white is close to zero. Exact values are in the table.")
    for i, (W_, b_) in enumerate(zip(run["W"], run["b"]), start=1):
        labels = run["prep"].feature_names_out if i == 1 else [f"neuron {j + 1}" for j in range(W_.shape[0])]
        with st.expander(f"Layer {i}: W {W_.shape}, b {b_.shape}, activation = {run['activations'][i - 1]}", expanded=(i == 1)):
            st.plotly_chart(weights_fig(W_, labels), width="stretch", theme=None, config={"displayModeBar": False})
            st.dataframe(pd.DataFrame(W_, index=labels), width="stretch")
            st.dataframe(pd.DataFrame(b_, index=["bias"]), width="stretch")

with tab_try:
    prep_, Xte_df_, yte_ = run["prep"], run["Xte_df"], run["yte"]
    cols_all = list(Xte_df_.columns)
    true_label = None
    if len(cols_all) <= 10:
        st.write("Enter your own input values:")
        ui = st.columns(min(len(cols_all), 4))
        row = {}
        for i, c in enumerate(cols_all):
            key = f"in_{run['dataset']}_{c}"
            if c in prep_.num_cols:
                row[c] = ui[i % len(ui)].number_input(str(c), value=float(prep_.medians[c]), format="%.4f", key=key)
            else:
                row[c] = ui[i % len(ui)].selectbox(str(c), prep_.categories[c], key=key)
        x_df = pd.DataFrame([row])
    else:
        st.write(f"Typing {len(cols_all)} features by hand is impractical, so pick a test sample instead:")
        idx = st.number_input("Test sample index", 0, len(Xte_df_) - 1, 0, key="sample_idx")
        x_df = Xte_df_.iloc[[int(idx)]]
        true_label = int(yte_[int(idx)])

    if st.button("Predict", key="predict", type="primary"):
        out = predict(prep_.transform(x_df), run["W"], run["b"], run["activations"])
        if run["n_classes"] == 2:
            p = float(out[0, 0])
            names = run["class_names"]
            st.success(f"Predicted class: **{names[int(p > 0.5)]}**  (P({names[1]}) = {p:.4f})")
        else:
            probs = out[0]
            st.success(f"Predicted class: **{run['class_names'][int(probs.argmax())]}**")
            st.bar_chart(pd.Series(probs, index=run["class_names"], name="probability"))
        if true_label is not None:
            st.caption(f"True label: {run['class_names'][true_label]}")