from hypothesis import given
from hypothesis.strategies import characters, integers, one_of, recursive, text, tuples

from plot import _Parser, _Tokenizer


def ends_with_repeat(x):
    return x.endswith(("+", "*", "}"))


def if_not_repeat(producer):
    return lambda x: x if ends_with_repeat(x) else producer(x)


def repeat(inner):
    def concat(start_stop):
        (start, stop) = sorted(start_stop)
        return inner.map(if_not_repeat(lambda x: f"{x}{{{start},{stop}}}"))

    n = integers(min_value=1, max_value=2 ** 31)
    return tuples(n, n).flatmap(concat)


def kleene(inner):
    return inner.map(if_not_repeat(lambda x: x + "*"))


def kleene1(inner):
    return inner.map(if_not_repeat(lambda x: x + "+"))


atom = one_of(
    text(".", min_size=1),
    text(characters(whitelist_categories={"Ll", "Lu"}), min_size=1),
)
regex = recursive(
    atom,
    lambda inner: one_of(
        inner.map(lambda x: f"({x})"),
        inner.flatmap(lambda x: inner.map(lambda y: f"{x}|{y}")),
        repeat(inner),
        kleene(inner),
        kleene1(inner),
        inner,
    ),
)


@given(regex)
def test_parse(pattern):
    parser = _Parser(_Tokenizer(pattern, 0))
    assert parser.parse()
