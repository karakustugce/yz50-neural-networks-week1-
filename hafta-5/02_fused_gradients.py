"""Exercise 2-3: cross entropy ve BatchNorm gradient'lerini tek ifadeye indir."""

from ortak import (
    comparison,
    detailed_forward,
    fused_parameter_grads,
    get_splits,
    init_parameters,
    make_batch,
    print_comparisons,
    retain_and_backward,
    save_result,
)

Xtr, Ytr = get_splits()["train"]
Xb, Yb = make_batch(Xtr, Ytr)
parameters = init_parameters()
loss, values = detailed_forward(parameters, Xb, Yb)
retain_and_backward(loss, values, parameters)
fused = fused_parameter_grads(parameters, values, Xb, Yb)

rows = [
    comparison("cross_entropy", fused["logits"], values["logits"]),
    comparison("batchnorm", fused["hprebn"], values["hprebn"]),
]
print_comparisons(rows)

failed = [row["name"] for row in rows if not row["approximate"]]
assert not failed, f"Uyusmayan kapali form gradient'leri: {failed}"

save_result(
    "exercise_2_3",
    {
        row["name"]: {
            "exact": row["exact"],
            "approximate": row["approximate"],
            "maxdiff": row["maxdiff"],
        }
        for row in rows
    },
)
print("\nSONUC: Cross entropy ve BatchNorm kapali formlari dogrulandi.")
