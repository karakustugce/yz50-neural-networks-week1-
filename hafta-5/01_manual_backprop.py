"""Exercise 1: butun ara degiskenlerin gradient'ini elle hesapla ve dogrula."""

from ortak import (
    comparison,
    detailed_forward,
    get_splits,
    init_parameters,
    make_batch,
    manual_backward,
    print_comparisons,
    retain_and_backward,
    save_result,
)

data = get_splits()
Xtr, Ytr = data["train"]
Xb, Yb = make_batch(Xtr, Ytr)
parameters = init_parameters()

loss, values = detailed_forward(parameters, Xb, Yb)
retain_and_backward(loss, values, parameters)
manual = manual_backward(parameters, values, Xb, Yb)

rows = []
for name, gradient in manual.items():
    target = parameters[name] if name in parameters else values[name]
    rows.append(comparison(name, gradient, target))

print(f"loss: {loss.item():.6f}\n")
print_comparisons(rows)

failed = [row["name"] for row in rows if not row["approximate"]]
assert not failed, f"Uyusmayan gradient'ler: {failed}"

save_result(
    "exercise_1",
    {
        "loss": round(loss.item(), 6),
        "gradient_sayisi": len(rows),
        "exact": sum(row["exact"] for row in rows),
        "approximate": sum(row["approximate"] for row in rows),
        "en_buyuk_fark": max(row["maxdiff"] for row in rows),
    },
)
print(f"\nSONUC: {len(rows)}/{len(rows)} gradient dogrulandi.")
