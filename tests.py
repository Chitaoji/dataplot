from src import dataplot as dp


def test_auto_label_for_multiple_data_calls_on_same_line():
    a, b, c = dp.data([]), dp.data([]), dp.data([])
    assert a.label == "a"
    assert b.label == "b"
    assert c.label == "c"

    a, b, c = (
        dp.data([]),
        dp.data([]),
        dp.data([]),
    )
    assert a.label == "a"
    assert b.label == "b"
    assert c.label == "c"


test_auto_label_for_multiple_data_calls_on_same_line()
