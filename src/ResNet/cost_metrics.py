"""
cost_metrics.py - Độ đo cost-sensitive cho bài toán phân loại mệnh giá tiền polymer VN.

Quy ước (theo Chung et al. 2017 và Lin 2014): C[y, k] = chi phí khi mẫu thật lớp y bị dự đoán là lớp k.
  - Hàng = lớp thật, cột = lớp dự đoán, đường chéo = 0.
  - Elkan (2001) viết ngược lại (hàng = dự đoán); ở đây dùng C[thật, dự đoán].
Nhãn là số nguyên 0..5 theo thứ tự DENOMS.

Dùng được cho mọi mô hình cho ra xác suất P (N x K): kNN (predict_proba),
softmax regression, CNN, ResNet (softmax của logits). Chỉ cần numpy.
"""
import sys
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DENOMS = (10_000, 20_000, 50_000, 100_000, 200_000, 500_000)
NAMES = tuple(str(d) for d in DENOMS)


# ----------------------------------------------------------------------
# 1. Xây dựng và kiểm tra ma trận chi phí
# ----------------------------------------------------------------------
def _normalize(C, floor=0.0):
    """Đưa trung bình các ô ngoài đường chéo về 1. Quyết định tối ưu không đổi khi
    nhân cả ma trận với một hằng số dương (Elkan 2001), nên chuẩn hóa là an toàn."""
    C = np.array(C, float)
    K = len(C)
    off = ~np.eye(K, dtype=bool)
    C[off] /= C[off].mean()
    if floor > 0:
        C[off] = np.maximum(C[off], floor)
        C[off] /= C[off].mean()
    np.fill_diagonal(C, 0.0)
    return C


def value_cost_matrix(denoms=DENOMS, mode="log", under=1.0, over=1.0, floor=0.0):
    """Chi phí theo chênh lệch giá trị tiền.
    mode: "abs"  -> |v_thật - v_dự đoán|  (thiệt hại tiền thật, 500k->10k rất nặng)
          "log"  -> |ln(v_thật / v_dự đoán)| (nhầm theo tỷ lệ, cân bằng hơn)
          "rank" -> |chênh lệch thứ hạng| (nhầm lân cận nhẹ, nhầm xa nặng)
    under: nhân thêm khi dự đoán THẤP hơn thật (đọc 500k thành 50k)
    over : nhân thêm khi dự đoán CAO hơn thật (đọc 50k thành 500k)
    floor: chi phí tối thiểu của mọi lỗi sau chuẩn hóa (vd 0.2)."""
    v = np.asarray(denoms, float)
    K = len(v)
    C = np.zeros((K, K))
    for y in range(K):
        for k in range(K):
            if y == k:
                continue
            if mode == "abs":
                d = abs(v[y] - v[k])
            elif mode == "log":
                d = abs(np.log(v[y] / v[k]))
            elif mode == "rank":
                d = abs(y - k)
            else:
                raise ValueError("mode phải là abs / log / rank")
            C[y, k] = d * (under if k < y else over)
    return _normalize(C, floor)


def with_class_importance(C, importance, normalize=True):
    """Nhân hàng của các lớp quan trọng. importance = {chỉ_số_lớp: hệ_số}.
    Vd {5: 3.0, 4: 2.0} nghĩa là sai khi tờ thật là 500k nặng gấp 3 lần."""
    C = np.array(C, float)
    for c, m in importance.items():
        C[c, :] *= m
    return _normalize(C) if normalize else C


def increasing_importance_matrix(denoms=DENOMS, alpha=0.5, base_mode="log", symmetric=False, floor=0.2):
    """Mệnh giá càng cao càng phải chính xác.
    symmetric=False (mặc định): nhân hàng của lớp thật y với (v_y / v_min) ** alpha,
        tức sai khi tờ thật là tiền lớn thì đắt hơn.
    symmetric=True: nhân theo (max(v_thật, v_dự đoán) / v_min) ** alpha,
        tức mọi lỗi có liên quan tới một tờ tiền lớn (thật hoặc bị đoán ra) đều đắt hơn.
    alpha: 0 = mọi lớp như nhau; 0.5 = căn bậc hai giá trị; 1 = tỷ lệ thẳng với giá trị."""
    v = np.asarray(denoms, float)
    base = value_cost_matrix(denoms, mode=base_mode, floor=floor)
    if symmetric:
        C = base * (np.maximum.outer(v, v) / v.min()) ** alpha
    else:
        C = base * ((v / v.min()) ** alpha)[:, None]
    return _normalize(C)


def check_cost_matrix(C):
    """Kiểm tra điều kiện hợp lý của Elkan: đường chéo 0, lỗi > 0, không có lớp
    nào luôn bị 'dominate' (khi đó lớp đó không bao giờ được dự đoán). Trả về list cảnh báo."""
    C = np.asarray(C, float)
    K = len(C)
    problems = []
    if not np.allclose(np.diag(C), 0):
        problems.append("Đường chéo phải bằng 0 (dự đoán đúng không tốn chi phí).")
    if (C[~np.eye(K, dtype=bool)] <= 0).any():
        problems.append("Mọi ô ngoài đường chéo phải > 0.")
    for m in range(K):
        for n in range(K):
            if m != n and np.all(C[:, m] >= C[:, n]):
                problems.append(f"Dự đoán lớp {m} luôn tốn không ít hơn lớp {n} -> lớp {m} không bao giờ được chọn.")
    return problems


# ----------------------------------------------------------------------
# 2. Quy tắc quyết định (Elkan 2001, công thức 1) và trọng số huấn luyện
# ----------------------------------------------------------------------
def bayes_predict(P, C):
    """Chọn lớp có chi phí kỳ vọng nhỏ nhất: argmin_k sum_y P(y|x) C[y,k].
    Không cần huấn luyện lại khi đổi ma trận chi phí."""
    return np.argmin(np.asarray(P) @ np.asarray(C), axis=1)


def class_weights_from_cost(C, class_counts=None):
    """Trọng số theo lớp thật cho weighted cross-entropy (Dmochowski et al. 2010):
    w_y = trung bình chi phí lỗi của lớp y. Đây chỉ là xấp xỉ theo hàng của ma trận.
    Nếu có class_counts thì chuẩn hóa để loss trung bình giữ cùng thang đo.
    PyTorch: nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))"""
    C = np.asarray(C, float)
    K = len(C)
    w = np.array([C[y, np.arange(K) != y].mean() for y in range(K)])
    if class_counts is not None:
        n = np.asarray(class_counts, float)
        w = w / ((w * n).sum() / n.sum())
    else:
        w = w / w.mean()
    return w


# ----------------------------------------------------------------------
# 3. Độ đo
# ----------------------------------------------------------------------
def confusion_matrix(y, pred, K=6):
    M = np.zeros((K, K), int)
    np.add.at(M, (np.asarray(y), np.asarray(pred)), 1)
    return M


def _div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)


def sample_costs(y, pred, C):
    """Chi phí từng mẫu: dùng cho bootstrap và so sánh cặp."""
    return np.asarray(C)[np.asarray(y), np.asarray(pred)]


def trivial_cost(y, C):
    """Chi phí của bộ phân loại hằng số tốt nhất (luôn đoán cùng một lớp).
    Gọi là 'trivial risk' trong Dmochowski et al. 2010."""
    C = np.asarray(C)
    return min(C[np.asarray(y), k].mean() for k in range(len(C)))


def evaluate(y, pred, C, class_names=NAMES, critical=None, severe_q=0.75):
    """Bộ độ đo đầy đủ cho một tập dự đoán nhãn.
    critical: danh sách chỉ số lớp quan trọng (để báo recall riêng).
    severe_q: lỗi 'nghiêm trọng' = lỗi có chi phí >= phân vị này của các ô ngoài đường chéo."""
    y, pred, C = np.asarray(y), np.asarray(pred), np.asarray(C, float)
    K = len(C)
    M = confusion_matrix(y, pred, K)
    tp = np.diag(M)
    recall = _div(tp, M.sum(1))
    precision = _div(tp, M.sum(0))
    f1 = _div(2 * precision * recall, precision + recall)

    costs = sample_costs(y, pred, C)
    avg_cost = costs.mean()
    triv = trivial_cost(y, C)
    thr = np.quantile(C[~np.eye(K, dtype=bool)], severe_q)
    cost_by_true = np.array([costs[y == c].sum() for c in range(K)])

    w_cls = np.array([C[c, np.arange(K) != c].mean() for c in range(K)])  # độ quan trọng từng lớp thật
    out = {
        "weighted_recall": float((w_cls * recall).sum() / w_cls.sum()),  # recall trung bình có trọng số
        "accuracy": float((y == pred).mean()),
        "macro_f1": float(f1.mean()),
        "min_recall": float(recall.min()),
        "avg_cost": float(avg_cost),                 # độ đo chính (Chung 2017, Lin 2014)
        "trivial_cost": float(triv),
        "normalized_cost": float(avg_cost / triv),   # < 1: tốt hơn đoán hằng số; càng nhỏ càng tốt
        "severe_error_rate": float((costs >= thr).mean()),
        "per_class": {class_names[c]: {"recall": float(recall[c]), "precision": float(precision[c]),
                                       "f1": float(f1[c]), "n": int(M[c].sum()),
                                       "cost_share": float(cost_by_true[c] / max(cost_by_true.sum(), 1e-12))}
                      for c in range(K)},
        "confusion": M,
        "cost_confusion": M * C,                     # ô nào đóng góp nhiều chi phí nhất
    }
    if critical is not None:
        crit = list(critical)
        out["critical_recall"] = float(tp[crit].sum() / max(M[crit].sum(), 1))
    return out


def evaluate_probs(y, P, C, **kw):
    """So sánh hai cách ra quyết định trên cùng xác suất P:
       'argmax' = quyết định thường, 'bayes' = quyết định theo chi phí kỳ vọng nhỏ nhất."""
    return {"argmax": evaluate(y, np.argmax(P, 1), C, **kw),
            "bayes": evaluate(y, bayes_predict(P, C), C, **kw)}


# ----------------------------------------------------------------------
# 4. Độ tin cậy xác suất (bước Bayes cần P được hiệu chỉnh tốt)
# ----------------------------------------------------------------------
def softmax(z, T=1.0):
    z = np.asarray(z, float) / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(P, y):
    return float(-np.log(np.clip(np.asarray(P)[np.arange(len(y)), y], 1e-12, 1)).mean())


def fit_temperature(logits_dev, y_dev, grid=np.linspace(0.05, 10, 400)):
    """Chọn nhiệt độ T làm NLL nhỏ nhất trên tập DEV (không dùng test)."""
    return float(grid[int(np.argmin([nll(softmax(logits_dev, T), y_dev) for T in grid]))])


def ece(P, y, bins=15):
    """Expected Calibration Error."""
    P, y = np.asarray(P), np.asarray(y)
    conf, pred = P.max(1), P.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


# ----------------------------------------------------------------------
# 5. Khoảng tin cậy và so sánh mô hình
# ----------------------------------------------------------------------
def bootstrap_ci(costs, n=2000, alpha=0.05, seed=0):
    """Trung bình chi phí và khoảng tin cậy bootstrap (mean, lo, hi)."""
    costs = np.asarray(costs)
    rng = np.random.default_rng(seed)
    means = costs[rng.integers(0, len(costs), (n, len(costs)))].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(costs.mean()), float(lo), float(hi)


def paired_bootstrap_diff(costs_a, costs_b, n=2000, alpha=0.05, seed=0):
    """Chênh lệch chi phí A - B trên cùng tập test (mean, lo, hi).
    Nếu hi < 0 thì A rẻ hơn B có ý nghĩa; nếu lo > 0 thì B rẻ hơn A."""
    d = np.asarray(costs_a) - np.asarray(costs_b)
    return bootstrap_ci(d, n, alpha, seed)


def importance_sweep(y, pred, C, critical, multipliers=(1, 2, 5, 10)):
    """Chi phí trung bình khi tăng dần độ quan trọng của các lớp critical
    (giống việc quét pcf trong Dmochowski 2010). Không chuẩn hóa, để so sánh giữa các mô hình."""
    y, pred = np.asarray(y), np.asarray(pred)
    return {m: float(with_class_importance(C, {c: m for c in critical}, normalize=False)[y, pred].mean())
            for m in multipliers}


# ----------------------------------------------------------------------
# 6. In báo cáo
# ----------------------------------------------------------------------
def report(res, title=""):
    print(f"=== {title} ===")
    print(f"accuracy={res['accuracy']:.4f}  macro_f1={res['macro_f1']:.4f}  min_recall={res['min_recall']:.4f}")
    print(f"avg_cost={res['avg_cost']:.4f}  trivial={res['trivial_cost']:.4f}  "
          f"normalized_cost={res['normalized_cost']:.4f}  severe_err={res['severe_error_rate']:.4f}")
    if "critical_recall" in res:
        print(f"critical_recall={res['critical_recall']:.4f}")
    print(f"{'lớp':>8} {'recall':>7} {'prec':>7} {'f1':>7} {'n':>5} {'cost%':>7}")
    for k, v in res["per_class"].items():
        print(f"{k:>8} {v['recall']:7.3f} {v['precision']:7.3f} {v['f1']:7.3f} {v['n']:5d} {100*v['cost_share']:6.1f}%")


# ----------------------------------------------------------------------
# 7. Bộ độ đo khuyến nghị: so sánh nhiều mô hình trong một lần gọi
# ----------------------------------------------------------------------
DEFAULT_ALPHA = 0.5
CRITICAL = (3, 4, 5)   # 100k, 200k, 500k


def default_cost_matrix(alpha=DEFAULT_ALPHA, symmetric=False):
    """Ma trận chi phí khuyến nghị: mệnh giá càng cao càng phải chính xác."""
    return increasing_importance_matrix(alpha=alpha, symmetric=symmetric)


def compare_models(y, probs, C=None, baseline=None, critical=CRITICAL,
                   decisions=("argmax", "bayes"), n_boot=2000, verbose=True):
    """So sánh nhiều mô hình trên cùng một tập (dev hoặc test).
    probs   : dict {tên: P}, P là ma trận xác suất N x K.
    baseline: khóa dạng "tên/argmax" để tính chênh lệch chi phí có khoảng tin cậy.
    Trả về list dict (mỗi dòng là một mô hình + một cách quyết định)."""
    C = default_cost_matrix() if C is None else np.asarray(C, float)
    y = np.asarray(y)
    rows, costs = [], {}
    for name, P in probs.items():
        for dec in decisions:
            pred = np.argmax(P, 1) if dec == "argmax" else bayes_predict(P, C)
            key = f"{name}/{dec}"
            r = evaluate(y, pred, C, critical=critical)
            c = sample_costs(y, pred, C)
            m, lo, hi = bootstrap_ci(c, n_boot)
            costs[key] = c
            rows.append({"model": key, "avg_cost": m, "ci_lo": lo, "ci_hi": hi,
                         "normalized_cost": r["normalized_cost"],
                         "weighted_recall": r["weighted_recall"],
                         "critical_recall": r.get("critical_recall", float("nan")),
                         "macro_f1": r["macro_f1"], "accuracy": r["accuracy"],
                         "severe_error_rate": r["severe_error_rate"], "pred": pred})
    if baseline is not None:
        for row in rows:
            row["delta_vs_baseline"] = paired_bootstrap_diff(costs[row["model"]], costs[baseline], n_boot)
    if verbose:
        _print_table(rows, baseline)
    return rows


def _print_table(rows, baseline):
    w = max(len(r["model"]) for r in rows) + 1
    head = (f"{'model':<{w}} {'avg_cost [CI95]':>24} {'norm':>6} {'w_rec':>6} {'crit':>6} "
            f"{'mF1':>6} {'acc':>6} {'sev':>6}")
    if baseline:
        head += f"  {'Δcost vs ' + baseline:<34}"
    print(head)
    for r in rows:
        line = (f"{r['model']:<{w}} {r['avg_cost']:7.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}] "
                f"{r['normalized_cost']:6.3f} {r['weighted_recall']:6.3f} {r['critical_recall']:6.3f} "
                f"{r['macro_f1']:6.3f} {r['accuracy']:6.3f} {r['severe_error_rate']:6.3f}")
        if baseline:
            d, lo, hi = r["delta_vs_baseline"]
            star = "*" if (hi < 0 or lo > 0) and r["model"] != baseline else ""
            line += f"  {d:+.4f} [{lo:+.4f},{hi:+.4f}]{star}"
        print(line)
    if baseline:
        print("* = khác baseline có ý nghĩa (khoảng tin cậy 95% không chứa 0). Δ âm = rẻ hơn baseline.")


def sensitivity(y, preds, alphas=(0.0, 0.25, 0.5, 1.0), symmetric=(False, True), verbose=True):
    """Kiểm tra xem thứ hạng mô hình có đổi khi đổi ma trận chi phí hay không.
    preds: dict {tên: nhãn dự đoán}. Dự đoán giữ nguyên, chỉ đổi ma trận dùng để chấm điểm.
    Trả về {(alpha, symmetric): {tên: avg_cost}}."""
    y = np.asarray(y)
    out = {}
    for sym in symmetric:
        for a in alphas:
            C = increasing_importance_matrix(alpha=a, symmetric=sym)
            out[(a, sym)] = {n: float(sample_costs(y, p, C).mean()) for n, p in preds.items()}
    if verbose:
        best_set = set()
        for (a, sym), d in out.items():
            order = sorted(d, key=d.get)
            best_set.add(order[0])
            print(f"alpha={a:<4} symmetric={str(sym):<5} -> " + " < ".join(f"{n} ({d[n]:.3f})" for n in order))
        print("Mô hình tốt nhất ổn định ở mọi cấu hình." if len(best_set) == 1
              else f"Cảnh báo: mô hình tốt nhất thay đổi theo ma trận chi phí: {sorted(best_set)}")
    return out


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Demo với dữ liệu giả lập (thay bằng P, y thật của bạn)
    rng = np.random.default_rng(0)
    counts = [281, 224, 223, 247, 317, 261]            # phân bố tập test trong tài liệu dự án
    y = np.repeat(np.arange(6), counts)
    P_knn = softmax(np.eye(6)[y] * 2.0 + rng.normal(0, 1.5, (len(y), 6)))
    P_resnet = softmax(np.eye(6)[y] * 4.0 + rng.normal(0, 1.5, (len(y), 6)))

    C = default_cost_matrix()
    print("Cảnh báo ma trận chi phí:", check_cost_matrix(C) or "không có")
    print("Class weights cho loss:", np.round(class_weights_from_cost(C, counts), 3), "\n")

    rows = compare_models(y, {"knn": P_knn, "resnet": P_resnet}, C, baseline="knn/argmax")
    print()
    sensitivity(y, {r["model"]: r["pred"] for r in rows})
    print()
    report(evaluate(y, rows[-1]["pred"], C, critical=CRITICAL), rows[-1]["model"])
