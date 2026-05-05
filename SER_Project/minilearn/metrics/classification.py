import numpy as np


def accuracy_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    label_idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_idx and p in label_idx:
            cm[label_idx[t], label_idx[p]] += 1
    return cm


def precision_score(y_true, y_pred, average='macro'):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    for lbl in labels:
        tp = np.sum((y_pred == lbl) & (y_true == lbl))
        fp = np.sum((y_pred == lbl) & (y_true != lbl))
        scores.append(float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0)
    return float(np.mean(scores)) if average == 'macro' else scores


def recall_score(y_true, y_pred, average='macro'):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    scores = []
    for lbl in labels:
        tp = np.sum((y_pred == lbl) & (y_true == lbl))
        fn = np.sum((y_pred != lbl) & (y_true == lbl))
        scores.append(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0)
    return float(np.mean(scores)) if average == 'macro' else scores


def f1_score(y_true, y_pred, average='macro'):
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)
    scores = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(prec, rec)]
    return float(np.mean(scores)) if average == 'macro' else scores


def classification_report(y_true, y_pred, labels=None, target_names=None):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    if target_names is None:
        target_names = [str(lbl) for lbl in labels]

    lines = [f"{'':>15}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>9}", '']
    for i, (name, p, r, f) in enumerate(zip(target_names, prec, rec, f1)):
        support = int(np.sum(y_true == labels[i]))
        lines.append(f"{name:>15}  {p:>9.2f}  {r:>9.2f}  {f:>9.2f}  {support:>9}")
    lines.append('')
    acc = accuracy_score(y_true, y_pred)
    lines.append(f"{'accuracy':>15}  {'':>9}  {'':>9}  {acc:>9.2f}  {len(y_true):>9}")
    lines.append(f"{'macro avg':>15}  {np.mean(prec):>9.2f}  {np.mean(rec):>9.2f}  {np.mean(f1):>9.2f}  {len(y_true):>9}")
    return '\n'.join(lines)
