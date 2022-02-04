from hypothesis import given
from hypothesis.strategies import (
    characters,
    integers,
    one_of,
    recursive,
    sampled_from,
    text,
    tuples,
)

from plot import _Parser, _Tokenizer, plot_re


def ends_with_repeat(x):
    return x.endswith(("+", "*", "}"))


def ends_with_at(x):
    return x.endswith(("^", "$", "\\A", "\\b", "\\B", "\\Z"))


def if_not_repeat_or_at(producer):
    return lambda x: x if ends_with_repeat(x) or ends_with_at(x) else producer(x)


def repeat(inner):
    def concat(start_stop):
        (start, stop) = sorted(start_stop)
        return inner.map(if_not_repeat_or_at(lambda x: f"{x}{{{start},{stop}}}"))

    n = integers(min_value=1, max_value=2 ** 31)
    return tuples(n, n).flatmap(concat)


def kleene(inner):
    return inner.map(if_not_repeat_or_at(lambda x: x + "*"))


def kleene1(inner):
    return inner.map(if_not_repeat_or_at(lambda x: x + "+"))


atom = one_of(
    text(".", min_size=1),
    text(characters(whitelist_categories={"Ll", "Lu"}), min_size=1),
    sampled_from(["^", "$", "\\A", "\\b", "\\B", "\\Z"]),
)
regex = recursive(
    atom,
    lambda inner: one_of(
        inner.map(lambda x: f"({x})"),
        inner.map(lambda x: f"(?:{x})"),
        inner.map(lambda x: f"(?!{x})"),
        inner.map(lambda x: f"(?={x})"),
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


@given(regex)
def test_plot(pattern):
    # Should not raise an exception
    plot_re(pattern)
