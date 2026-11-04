import numpy as np
from sklearn.metrics import confusion_matrix as sk_cm, precision_recall_fscore_support
from metrics import classification_report, confusion_matrix

rng = np.random.default_rng(0)
for n in (2, 3, 10):
    y = rng.integers(0, n, 300)
    p = np.where(rng.random(300) < 0.7, y, rng.integers(0, n, 300))
    cm = confusion_matrix(y, p, n)
    assert (cm == sk_cm(y, p, labels=range(n))).all()
    r = classification_report(cm)
    P, R, F, S = precision_recall_fscore_support(y, p, labels=range(n), zero_division=0)
    assert np.allclose(r["precision"], P) and np.allclose(r["recall"], R) and np.allclose(r["f1"], F) and (r["support"] == S).all()
    assert np.isclose(r["macro"]["f1"], F.mean())
cm = confusion_matrix([0, 0, 1, 1], [0, 0, 0, 0], 2)
assert classification_report(cm)["precision"][1] == 0
print("metrics OK")
