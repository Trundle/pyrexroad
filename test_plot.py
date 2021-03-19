from hypothesis import given
from hypothesis.strategies import characters, one_of, recursive, text

from plot import _Parser, _Tokenizer


atom = one_of(
    text(".", min_size=1),
    text(characters(whitelist_categories={"Ll", "Lu"}), min_size=1),
)
regex = recursive(
    atom,
    lambda inner: one_of(
        inner.map(lambda x: f"({x})"),
        inner.flatmap(lambda x: inner.map(lambda y: f"{x}|{y}")),
        inner,
        # XXX add kleene
    ),
)


@given(regex)
def test_parse(pattern):
    print(pattern)
    parser = _Parser(_Tokenizer(pattern, 0))
    assert parser.parse()
