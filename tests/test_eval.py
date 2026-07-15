"""Offline checks for the evaluator (no model / no download needed).

Run:  python -m tests.test_eval
"""
from shared.eval import evaluate, flatten, build_comparison_table

GOLD = {
    "menu": [
        {"nm": "Nasi Campur Bali", "cnt": "1 x", "price": "75,000"},
        {"nm": "Ice Lemon Tea", "cnt": "1 x", "price": "24,000"},
    ],
    "sub_total": {"subtotal_price": "99,000", "tax_price": "9,900"},
    "total": {"total_price": "108,900"},
}


def test_perfect_match():
    res = evaluate([GOLD], [GOLD], domain="cord")
    assert res["micro_f1"] == 1.0, res["report"]
    assert res["json_valid_rate"] == 1.0
    print("[ok] perfect match -> F1=1.0")


def test_normalization_saves_score():
    # total_price "108900" vs gold "108,900" must match after normalization
    pred = {**GOLD, "total": {"total_price": "108900"}}
    res = evaluate([pred], [GOLD], domain="cord")
    assert res["micro_f1"] == 1.0, "normalization failed: " + res["report"]
    print("[ok] '108900' == '108,900' after normalize -> F1=1.0")


def test_missing_field_is_fn():
    # drop tax_price -> exactly one FN, no FP
    pred = {**GOLD, "sub_total": {"subtotal_price": "99,000"}}
    res = evaluate([pred], [GOLD], domain="cord")
    tax = res["per_field"]["sub_total.tax_price"]
    assert (tax["tp"], tax["fp"], tax["fn"]) == (0, 0, 1), tax
    assert res["micro_p"] == 1.0 and res["micro_r"] < 1.0
    print(f"[ok] missing tax_price -> FN=1, P=1.0, R={res['micro_r']:.3f}")


def test_wrong_value_is_fp_and_fn():
    # wrong price -> one FP (wrong) + one FN (missed correct)
    pred = {**GOLD, "total": {"total_price": "999,999"}}
    res = evaluate([pred], [GOLD], domain="cord")
    tp = res["per_field"]["total.total_price"]
    assert (tp["tp"], tp["fp"], tp["fn"]) == (0, 1, 1), tp
    print("[ok] wrong total_price -> FP=1 AND FN=1")


def test_parse_fail_penalized():
    res = evaluate(["这不是 JSON"], [GOLD], domain="cord")
    assert res["json_valid_rate"] == 0.0
    assert res["micro_f1"] == 0.0
    print("[ok] unparseable output -> validity 0, F1 0")


def test_list_order_independent():
    # reversed menu order must not matter (multiset, no index)
    pred = {**GOLD, "menu": list(reversed(GOLD["menu"]))}
    res = evaluate([pred], [GOLD], domain="cord")
    assert res["micro_f1"] == 1.0, "list order should not matter"
    print("[ok] menu order reversed -> F1=1.0 (order-independent)")


def test_fenced_raw_output():
    raw = '```json\n{"total": {"total_price": "108,900"}}\n```'
    res = evaluate([raw], [{"total": {"total_price": "108,900"}}], domain="cord")
    assert res["json_valid_rate"] == 1.0 and res["micro_f1"] == 1.0
    print("[ok] ```json fenced``` raw string parsed and scored")


if __name__ == "__main__":
    print("flatten(GOLD) =")
    for kv in flatten(GOLD):
        print("   ", kv)
    print()
    for fn in [
        test_perfect_match,
        test_normalization_saves_score,
        test_missing_field_is_fn,
        test_wrong_value_is_fp_and_fn,
        test_parse_fail_penalized,
        test_list_order_independent,
        test_fenced_raw_output,
    ]:
        fn()
    print("\nALL TESTS PASSED ✅")

    # demo the comparison-table renderer with two fake runs
    e0 = evaluate([{"total": {"total_price": "1"}}], [GOLD], domain="cord")
    e3 = evaluate([GOLD], [GOLD], domain="cord")
    print("\n" + build_comparison_table({"E0 基座·自由": e0, "E3 LoRA·约束": e3}))
