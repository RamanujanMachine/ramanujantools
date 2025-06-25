import pytest

from ramanujantools.utils import batched, Batchable


class Dummy:
    @batched("x")
    def process(self, x: Batchable[int]) -> Batchable[int]:
        return [i * 2 for i in x]


def test_scalar_input_returns_scalar():
    d = Dummy()
    result = d.process(5)
    assert result == 10  # single scalar return, unwrapped


def test_list_input_returns_list():
    d = Dummy()
    result = d.process([1, 2, 3])
    assert result == [2, 4, 6]


def test_invalid_scalar_type_raises():
    d = Dummy()
    with pytest.raises(TypeError):
        d.process("string_instead_of_int")


def test_invalid_list_element_type_raises():
    d = Dummy()
    with pytest.raises(TypeError):
        d.process([1, "bad", 3])


def test_missing_argument_raises():
    d = Dummy()
    with pytest.raises(TypeError):
        Dummy.process(d)  # no argument


def test_preserves_other_args():
    class Dummy2:
        @batched("y")
        def f(self, x: int, y: Batchable[int]) -> Batchable[int]:
            return [x + i for i in y]

    d = Dummy2()
    assert d.f(10, 5) == 15
    assert d.f(10, [1, 2]) == [11, 12]


def test_missing_annotation_raises():
    with pytest.raises(TypeError, match="No annotation for argument 'missing'"):

        class Bad:
            @batched("missing")
            def foo(self, x: int) -> None:
                pass


def test_invalid_annotation_format_raises():
    with pytest.raises(TypeError, match=r"annotation must be.*List"):

        class Bad2:
            @batched("x")
            def foo(self, x: int) -> None:
                pass


def test_argument_not_provided_raises():
    d = Dummy()
    # Cannot skip required argument without triggering Python's own TypeError,
    # so just test that TypeError (not our ValueError) is raised:
    with pytest.raises(TypeError):
        d.process()


def test_scalar_wrong_type_raises():
    d = Dummy()
    with pytest.raises(TypeError, match="Argument 'x' must be of type Batchable"):
        d.process("not an int")


def test_list_element_wrong_type_raises():
    d = Dummy()
    with pytest.raises(
        TypeError, match="All elements of argument 'x' must be of type Batchable"
    ):
        d.process([1, "bad", 3])
