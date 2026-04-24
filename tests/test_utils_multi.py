import unittest

from dataplot.utils.multi import (
    UNSUBSCRIPTABLE,
    MultiObject,
    cleaner,
    multiple,
    multipartial,
    single,
)


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


class TestMultiUtils(unittest.TestCase):
    def test_multiobject_getattr_and_call(self):
        mo = MultiObject([Box(1), Box(2)])

        values = mo.value
        added = mo.add(3)

        self.assertEqual(values.__multiobjects__, [1, 2])
        self.assertEqual(added.__multiobjects__, [4, 5])

    def test_multiobject_call_with_reflex(self):
        mo = MultiObject([F(1), F(2)], call_reflex=True)

        result = mo(3)

        self.assertEqual(result.__multiobjects__, [4, 9])

    def test_multipartial_single_multiple_and_cleaner(self):
        ctor = multipartial(call_reflex=False)
        mo = ctor([1, 2, 3])

        self.assertIsInstance(mo, MultiObject)
        self.assertEqual(single(mo, n=1), 2)
        self.assertEqual(single(99), 99)
        self.assertEqual(multiple(mo), [1, 2, 3])
        self.assertEqual(multiple(5), [5])
        self.assertIsNone(cleaner([None, None]))
        self.assertEqual(cleaner([None, 1]).__multiobjects__, [None, 1])

    def test_multiobject_binary_operations_for_scalars_and_multiobject(self):
        mo = MultiObject([1, 2, 3])
        self.assertEqual((mo + 2).__multiobjects__, [3, 4, 5])
        self.assertEqual((2 + mo).__multiobjects__, [3, 4, 5])

        mo2 = MultiObject([10, 20, 30])
        self.assertEqual((mo2 - mo).__multiobjects__, [9, 18, 27])

    def test_multiobject_getitem_behavior(self):
        seq = MultiObject([[1, 2], [3, 4]])
        self.assertEqual(seq[0].__multiobjects__, [1, 3])

        class MaybeSubscriptable:
            def __init__(self, value):
                self.value = value

            def __getitem__(self, key):
                if isinstance(key, int):
                    return UNSUBSCRIPTABLE
                return self.value[key]

        direct = MultiObject([MaybeSubscriptable("ab"), MaybeSubscriptable("cd")])
        self.assertEqual(direct[1].value, "cd")


if __name__ == "__main__":
    unittest.main()
