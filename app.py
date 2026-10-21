import numpy as np
import pandas as pd
import plotly.graph_objects as pgo
import streamlit as st
from sklearn.datasets import load_digits, load_iris, make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from network import predict, train

st.set_page_config(page_title="NN From Scratch", layout="wide")
st.title("🧠 Neural Network From Scratch")

HIDDEN_ACTIVATIONS = ["sigmoid", "tanh", "relu", "leaky_relu", "linear"]


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


def draw_network(layer_sizes, activations, W, max_show=10):
    """Neurons as circles, weights as lines: blue = positive, red = negative, thicker = bigger."""
    shown = [min(n, max_show) for n in layer_sizes]
    n_layers = len(layer_sizes)
    pos = []
    for l, n in enumerate(shown):
        ys = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) if n > 1 else np.array([0.0])
        pos.append((np.full(n, float(l)), ys))

    widths = (0.5, 1.5, 3.0)
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
            x=xs, y=ys, mode="lines", opacity=0.6, hoverinfo="skip", showlegend=False,
            line=dict(color="royalblue" if s == 1 else "crimson", width=widths[k]),
        ))
    notes = []
    for l in range(n_layers):
        fig.add_trace(pgo.Scatter(
            x=pos[l][0], y=pos[l][1], mode="markers", hoverinfo="skip", showlegend=False,
            marker=dict(size=14, color="white", line=dict(color="black", width=1.5)),
        ))
        name = "input" if l == 0 else activations[l - 1]
        extra = f"<br>(first {max_show} shown)" if layer_sizes[l] > max_show else ""
        notes.append(dict(x=l, y=pos[l][1].max() + 0.9, text=f"{layer_sizes[l]} neurons<br>{name}{extra}",
                          showarrow=False, font=dict(size=11)))
    fig.update_layout(height=480, annotations=notes, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def draw_boundary(run):
    """Colour map of what the trained network predicts at every point of the 2D plane."""
    Xtr_raw, Xte_raw, ytr, yte = run["Xtr_raw"], run["Xte_raw"], run["ytr"], run["yte"]
    Xall = np.vstack([Xtr_raw, Xte_raw])
    gx = np.linspace(Xall[:, 0].min() - 0.5, Xall[:, 0].max() + 0.5, 150)
    gy = np.linspace(Xall[:, 1].min() - 0.5, Xall[:, 1].max() + 0.5, 150)
    xx, yy = np.meshgrid(gx, gy)
    grid = run["scaler"].transform(np.c_[xx.ravel(), yy.ravel()])
    out = predict(grid, run["W"], run["b"], run["activations"])
    binary = run["n_classes"] == 2
    z = (out[:, 0] if binary else out.argmax(axis=1)).reshape(xx.shape)

    fig = pgo.Figure()
    fig.add_trace(pgo.Contour(
        x=gx, y=gy, z=z, opacity=0.55, showscale=binary, line_width=0,
        colorscale="RdBu" if binary else "Viridis",
        zmin=0 if binary else None, zmax=1 if binary else None,
        colorbar=dict(title="P(class 1)") if binary else None,
    ))
    palette = ["crimson", "royalblue", "green", "orange", "purple", "brown", "teal", "gray", "olive", "black"]
    for c in range(run["n_classes"]):
        col = palette[c % len(palette)]
        m = ytr == c
        fig.add_trace(pgo.Scatter(x=Xtr_raw[m, 0], y=Xtr_raw[m, 1], mode="markers", name=f"class {c} (train)",
                                  marker=dict(size=5, color=col, opacity=0.5)))
        m = yte == c
        fig.add_trace(pgo.Scatter(x=Xte_raw[m, 0], y=Xte_raw[m, 1], mode="markers", name=f"class {c} (test)",
                                  marker=dict(size=8, color=col, symbol="diamond", line=dict(color="black", width=1))))
    fig.update_layout(height=520, xaxis_title="feature 1", yaxis_title="feature 2", margin=dict(l=10, r=10, t=20, b=10))
    return fig


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Data")
    dataset = st.selectbox("Dataset", ["XOR", "Moons", "Circles", "Iris", "Digits"])
    test_size = st.slider("Test split", 0.1, 0.5, 0.2, 0.05)

    st.header("Architecture")
    n_hidden = st.number_input("Hidden layers", 0, 8, 2)
    sizes, acts = [], []
    for i in range(int(n_hidden)):
        c1, c2 = st.columns(2)
        sizes.append(c1.number_input(f"L{i + 1} neurons", 1, 512, 16, key=f"n{i}"))
        acts.append(c2.selectbox(f"L{i + 1} activation", HIDDEN_ACTIVATIONS, key=f"a{i}"))

    st.header("Training (Adam)")
    epochs = st.number_input("Epochs", 1, 5000, 100)
    batch_size = st.number_input("Batch size", 1, 4096, 32)
    lr = st.number_input("Learning rate", 0.00001, 1.0, 0.01, step=0.001, format="%.5f")
    go = st.button("🚀 Train", type="primary")

# ---------------- data + task ----------------
X, y = load_data(dataset)
n_classes = len(np.unique(y))
n_features = X.shape[1]

if n_classes == 2:
    n_out, out_act, loss_name = 1, "sigmoid", "bce"
else:
    n_out, out_act, loss_name = n_classes, "softmax", "cce"

layer_sizes = [n_features, *map(int, sizes), n_out]
activations = [*acts, out_act]

c1, c2, c3 = st.columns(3)
c1.metric("Samples", len(X))
c2.metric("Features", n_features)
c3.metric("Classes", n_classes)
st.write(f"**Layers:** `{' → '.join(map(str, layer_sizes))}`")
st.write(f"**Activations:** `{' → '.join(activations)}`  |  **Loss:** `{loss_name}`")

# ---------------- train ----------------
if go:
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

    if n_classes == 2:
        Ytr, Yte = ytr.reshape(-1, 1).astype(float), yte.reshape(-1, 1).astype(float)
    else:
        Ytr, Yte = np.eye(n_classes)[ytr], np.eye(n_classes)[yte]

    with st.spinner("Training..."):
        W, b, h = train(
            Xtr, Ytr, layer_sizes, activations, loss_name,
            epochs=int(epochs), batch_size=int(batch_size), lr=float(lr),
            X_val=Xte, y_val=Yte,
        )

    # Save results in session_state so they don't disappear when Predict triggers a rerun
    st.session_state["run"] = dict(
        W=W, b=b, h=h, scaler=scaler, activations=activations, dataset=dataset,
        n_classes=n_classes, Xte_raw=Xte_raw, yte=yte,
        Xtr_raw=Xtr_raw, ytr=ytr, layer_sizes=layer_sizes,
    )

run = st.session_state.get("run")
if run is None:
    st.info("Configure the network in the sidebar, then press **Train**.")
    st.stop()

h = run["h"]
st.subheader("Results")
st.caption(f"Last trained on: {run['dataset']}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Train loss", f"{h['loss'][-1]:.4f}")
m2.metric("Test loss", f"{h['val_loss'][-1]:.4f}")
m3.metric("Train accuracy", f"{h['score'][-1]:.4f}")
m4.metric("Test accuracy", f"{h['val_score'][-1]:.4f}")

tab_curves, tab_net, tab_boundary, tab_weights, tab_try = st.tabs(
    ["📉 Learning curves", "🕸️ Network", "🗺️ Decision boundary", "⚖️ Weights", "🎯 Try it"]
)

with tab_curves:
    ca, cb = st.columns(2)
    ca.caption("Loss")
    ca.line_chart(pd.DataFrame({"train": h["loss"], "test": h["val_loss"]}))
    cb.caption("Accuracy")
    cb.line_chart(pd.DataFrame({"train": h["score"], "test": h["val_score"]}))

with tab_net:
    biggest = max(run["layer_sizes"])
    if biggest <= 10:
        max_show = biggest
    else:
        max_show = st.slider("Max neurons shown per layer", 5, min(biggest, 128), min(biggest, 30))
    st.caption("Blue lines = positive weights, red = negative. Thicker = larger |weight|.")
    st.plotly_chart(
        draw_network(run["layer_sizes"], run["activations"], run["W"], max_show=max_show),
        width="stretch",
    )

with tab_boundary:
    if run["Xte_raw"].shape[1] == 2:
        st.caption("Background = network's prediction at that point. Dots = train, diamonds = test.")
        st.plotly_chart(draw_boundary(run), width="stretch")
    else:
        st.info("Decision boundaries are only available for datasets with 2 features (XOR, Moons, Circles).")

with tab_weights:
    for i, (W_, b_) in enumerate(zip(run["W"], run["b"]), start=1):
        with st.expander(f"Layer {i}: W {W_.shape}, b {b_.shape}, activation = {run['activations'][i - 1]}"):
            st.dataframe(pd.DataFrame(W_))
            st.dataframe(pd.DataFrame(b_, index=["bias"]))

with tab_try:
    Xte_raw, yte = run["Xte_raw"], run["yte"]
    n_feat = Xte_raw.shape[1]
    true_label = None
    if n_feat <= 10:
        st.write("Enter your own input values:")
        defaults = Xte_raw.mean(axis=0)
        cols = st.columns(min(n_feat, 4))
        vals = [
            cols[i % len(cols)].number_input(
                f"feature {i + 1}", value=float(defaults[i]), format="%.4f", key=f"in_{run['dataset']}_{i}"
            )
            for i in range(n_feat)
        ]
        x_raw = np.array(vals)
    else:
        st.write(f"Typing {n_feat} features by hand is impractical, so pick a test sample instead:")
        idx = st.number_input("Test sample index", 0, len(Xte_raw) - 1, 0)
        x_raw = Xte_raw[int(idx)]
        true_label = int(yte[int(idx)])

    if st.button("Predict"):
        x = run["scaler"].transform(x_raw.reshape(1, -1))
        out = predict(x, run["W"], run["b"], run["activations"])
        if run["n_classes"] == 2:
            p = float(out[0, 0])
            st.success(f"Predicted class: **{int(p > 0.5)}**  (P(class 1) = {p:.4f})")
        else:
            probs = out[0]
            st.success(f"Predicted class: **{int(probs.argmax())}**")
            st.bar_chart(pd.Series(probs, name="probability"))
        if true_label is not None:
            st.caption(f"True label: {true_label}")