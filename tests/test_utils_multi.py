from dataplot.utils.multi import MultiObject, cleaner, multiple, multipartial, single


class Box:
    def __init__(self, value):
        self.value = value

    def add(self, num):
        return self.value + num


class F:
    def __init__(self, offset=0):
        self.offset = offset

    def __call__(self, x, **kwargs):
        prev = kwargs.get("__multi_prev_returned__")
        return x + self.offset + (prev or 0)


def test_multiobject_getattr_and_call():
    mo = MultiObject([Box(1), Box(2)])

    values = mo.value
    added = mo.add(3)

    assert values.__multiobjects__ == [1, 2]
    assert added.__multiobjects__ == [4, 5]


def test_multiobject_call_with_reflex():
    mo = MultiObject([F(1), F(2)], call_reflex=True)

    result = mo(3)

    assert result.__multiobjects__ == [4, 9]


def test_multipartial_single_multiple_and_cleaner():
    ctor = multipartial(call_reflex=False)
    mo = ctor([1, 2, 3])

    assert isinstance(mo, MultiObject)
    assert single(mo, n=1) == 2
    assert single(99) == 99
    assert multiple(mo) == [1, 2, 3]
    assert multiple(5) == [5]
    assert cleaner([None, None]) is None
    assert cleaner([None, 1]).__multiobjects__ == [None, 1]
