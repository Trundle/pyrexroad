from __future__ import annotations

import sre_constants
import sre_parse
import sre_compile
from dataclasses import dataclass, field
from functools import singledispatchmethod
from itertools import groupby
from typing import Optional, Sequence

import railroad as rr


@dataclass
class _Token:
    pos: int
    value: Any


class _Tokenizer:
    """
    Converts a regex into op codes and argument tokens. Only accepts
    syntactically valid regexes as input.
    """

    class _Context:
        def __init__(self, tokenizer, max_pos):
            self.tokenizer = tokenizer
            self.max_pos = max_pos
            self.previous_max_pos = 0

        def __enter__(self):
            self.previous_max_pos = self.tokenizer.max_pos
            self.tokenizer.max_pos = self.max_pos
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.tokenizer.max_pos = self.previous_max_pos

    def __init__(self, pattern: str, flags=0):
        c = sre_compile._code  # type: ignore [attr-defined]
        self.tokens = c(sre_parse.parse(pattern, flags), flags)
        self.pos = 0
        self.max_pos = len(self.tokens)

    def peek_token(self):
        if self.pos < self.max_pos:
            return self.tokens[self.pos]

    def next_token(self) -> Optional[_Token]:
        if self.pos < self.max_pos:
            token = _Token(self.pos, self.tokens[self.pos])
            self.pos += 1
            return token
        return None

    def limit_to_next(self, n):
        return self._Context(self, self.pos + n)


## Regex AST


@dataclass
class Op:
    pos: int
    next: "Optional[Op]" = field(init=False, repr=False)

    def __post_init__(self):
        self.next = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass(eq=False)
class Any(Op):
    pass


@dataclass(eq=False)
class Assert(Op):
    skip: int
    back: int
    pattern: Op


@dataclass(eq=False)
class AssertNot(Op):
    skip: int
    back: int
    pattern: Op


@dataclass(eq=False)
class At(Op):
    where: int


@dataclass(eq=False)
class Branch(Op):
    branches: list[list[Op]]


@dataclass(eq=False)
class Category(Op):
    category: int


@dataclass(eq=False)
class Failure(Op):
    pass


@dataclass(eq=False)
class GroupRef(Op):
    group: int


@dataclass(eq=False)
class In(Op):
    skip: int
    negate: bool
    ranges: list[Op]


@dataclass(eq=False)
class Info(Op):
    skip: int
    flags: int
    min_width: int
    max_width: int


@dataclass(eq=False)
class Jump(Op):
    skip: int


@dataclass(eq=False)
class Literal(Op):
    literal: int


@dataclass(eq=False)
class Mark(Op):
    group: int


@dataclass(eq=False)
class NotLiteral(Op):
    literal: int


@dataclass(eq=False)
class Range(Op):
    min_char: int
    max_char: int


@dataclass(eq=False)
class Repeat(Op):
    minimal: bool
    skip: int
    min_times: int
    max_times: int
    body: list[Op]
    epilogue: int


@dataclass(eq=False)
class Success(Op):
    pass


@dataclass(eq=False)
class RepeatOne(Op):
    minimal: bool
    skip: int
    min_times: int
    max_times: int
    op: Op
    success: Success


## Parser


class _Parser:
    """
    Parses regex opcodes into an AST.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def parse(self) -> Sequence[Op]:
        ops = self.ops()
        if ops is None or self.tokenizer.peek_token():
            raise RuntimeError(
                f"Parsing error :( This is unexpected. Next token: {self.tokenizer.peek_token()}"
            )
        return ops

    def op(self):
        return self._alt(
            self.any,
            self.assert_,
            self.assert_not,
            self.at,
            self.branch,
            self.category,
            self.failure,
            self.group_ref,
            self.in_,
            self.info,
            self.jump,
            self.literal,
            self.mark,
            self.not_literal,
            self.range,
            self.repeat,
            self.repeat_one,
            self.success,
        )

    def ops(self) -> Optional[list[Op]]:
        return self._loop(self.op)

    def any(self) -> Optional[Any]:
        if op := self._expect(sre_constants.ANY):
            return Any(op.pos)
        return None

    def assert_(self):
        if op := self._expect(sre_constants.ASSERT):
            return self._assert_common(op, Assert)

    def assert_not(self):
        if op := self._expect(sre_constants.ASSERT_NOT):
            return self._assert_common(op, AssertNot)

    def _assert_common(self, op, result):
        skip = self.int()
        back = self.int()
        with self.tokenizer.limit_to_next(skip - 3):
            pattern = self.ops()
        if self._expect(sre_constants.SUCCESS):
            return result(op.pos, skip, back, pattern)

    def at(self):
        if op := self._expect(sre_constants.AT):

            def e(value):
                return lambda: self._expect(value)

            if (
                where := self._alt(
                    e(sre_constants.AT_BEGINNING),
                    e(sre_constants.AT_BEGINNING_LINE),
                    e(sre_constants.AT_BEGINNING_STRING),
                    e(sre_constants.AT_BOUNDARY),
                    e(sre_constants.AT_LOC_BOUNDARY),
                    e(sre_constants.AT_LOC_NON_BOUNDARY),
                    e(sre_constants.AT_NON_BOUNDARY),
                    e(sre_constants.AT_UNI_BOUNDARY),
                    e(sre_constants.AT_UNI_NON_BOUNDARY),
                    e(sre_constants.AT_END_LINE),
                    e(sre_constants.AT_END_STRING),
                    e(sre_constants.AT_END),
                )
            ) is not None:
                return At(op.pos, where.value)

    def branch(self):
        if op := self._expect(sre_constants.BRANCH):
            skip = self.int()
            branches = []
            while skip:
                with self.tokenizer.limit_to_next(skip - 1):
                    branch = self.ops()
                if branch is not None:
                    branches.append(branch)
                    skip = self.int()
                else:
                    break
            if skip is not None:
                return Branch(op.pos, branches)

    def category(self):
        if op := self._expect(sre_constants.CATEGORY):
            category = self.int()
            return Category(op.pos, category)

    def failure(self):
        if op := self._expect(sre_constants.FAILURE):
            return Failure(op.pos)

    def group_ref(self):
        if op := self._expect(sre_constants.GROUPREF):
            if (group := self.int()) is not None:
                return GroupRef(op.pos, group)

    def in_(self):
        if op := self._expect(sre_constants.IN):
            if (skip := self.int()) is not None:
                negate = False
                if self.tokenizer.peek_token() == sre_constants.NEGATE:
                    self.tokenizer.next_token()
                    negate = True
                with self.tokenizer.limit_to_next(skip - 2 - negate):
                    ranges = self.ops()
                if self._expect(sre_constants.FAILURE):
                    return In(op.pos, skip, negate, ranges)

    def info(self) -> Optional[Info]:
        if op := self._expect(sre_constants.INFO):
            (skip, flags, min_width, max_width) = self._ints(4)
            if max_width is not None:
                for _ in range(skip - 4):
                    self.tokenizer.next_token()
                return Info(op.pos, skip, flags, min_width, max_width)
        return None

    def jump(self):
        if op := self._expect(sre_constants.JUMP):
            if (skip := self.int()) is not None:
                return Jump(op.pos, skip)

    def literal(self):
        if op := self._expect(sre_constants.LITERAL):
            if (literal := self.int()) is not None:
                return Literal(op.pos, literal)

    def mark(self):
        if op := self._expect(sre_constants.MARK):
            if (group := self.int()) is not None:
                return Mark(op.pos, group)

    def not_literal(self):
        if op := self._expect(sre_constants.NOT_LITERAL):
            if (literal := self.int()) is not None:
                return NotLiteral(op.pos, literal)

    def range(self):
        if op := self._expect(sre_constants.RANGE):
            (min_char, max_char) = self._ints(2)
            if max_char is not None:
                return Range(op.pos, min_char, max_char)

    def repeat(self):
        if (op := self._expect(sre_constants.REPEAT)) or (
            op := self._expect(sre_constants.MIN_REPEAT)
        ):
            (skip, min_times, max_times) = self._ints(3)
            if max_times is not None:
                with self.tokenizer.limit_to_next(skip - 1):
                    if (body := self.ops()) is not None:

                        def constant(value):
                            def match():
                                if token := self._expect(value):
                                    return token.value

                            return match

                        epilogue = self._alt(
                            constant(sre_constants.MAX_UNTIL),
                            constant(sre_constants.MIN_UNTIL),
                        )
                        if epilogue is not None:
                            return Repeat(
                                op.pos,
                                op.value == sre_constants.MIN_REPEAT,
                                skip,
                                min_times,
                                max_times,
                                body,
                                epilogue,
                            )

    def repeat_one(self):
        if (op := self._expect(sre_constants.REPEAT_ONE)) or (
            op := self._expect(sre_constants.MIN_REPEAT_ONE)
        ):
            (skip, min_times, max_times) = self._ints(3)
            if max_times is not None:
                with self.tokenizer.limit_to_next(skip - 1):
                    if (loop_op := self.op()) is not None:
                        if success := self.success():
                            return RepeatOne(
                                op.pos,
                                op.value == sre_constants.MIN_REPEAT_ONE,
                                skip,
                                min_times,
                                max_times,
                                loop_op,
                                success,
                            )

    def success(self):
        if op := self._expect(sre_constants.SUCCESS):
            return Success(op.pos)

    def int(self) -> Optional[int]:
        if isinstance(self.tokenizer.peek_token(), int):
            return self.tokenizer.next_token().value
        return None

    def _ints(self, n):
        """
        Expect to match `n` ints next. If a match fails, returns `None` for the
        remaining matches so that it can be used in an unpacking expression.
        """
        for i in range(n):
            value = self.int()
            yield value
            if value is None:
                break
        for _ in range(i + 1, n):
            yield None

    def _alt(self, *branches):
        for branch in branches:
            if (match := branch()) is not None:
                return match

    def _expect(self, value):
        if self.tokenizer.peek_token() == value:
            return self.tokenizer.next_token()

    def _loop(self, f):
        nodes = []
        while (node := f()) is not None:
            nodes.append(node)
        if nodes:
            return nodes


## Plotting


@dataclass
class _Group:
    group: int
    ops: list[Op]


@dataclass
class _LiteralSeq:
    literals: list[Literal]


class _Plotter:
    def __init__(self):
        self._expected_jumps = []

    def plot(self, code):
        return self._visit_op_seq(code)

    def _visit_op_seq(self, code):
        nodes = []
        for op in self._groupify(self._simplify(code)):
            if (node := self._visit(op)) is not None:
                nodes.append(node)
        return rr.Sequence(*nodes)

    def _simplify(self, code):
        for (is_literal, ops) in groupby(code, key=lambda op: isinstance(op, Literal)):
            if is_literal:
                yield _LiteralSeq(list(ops))
            else:
                yield from ops

    def _groupify(self, code):
        code = list(code)
        marks = dict(
            (op.group, i) for (i, op) in enumerate(code) if isinstance(op, Mark)
        )
        if marks:
            (group, start) = next(iter(marks.items()))
            if group % 2:
                raise ValueError(f"Groups should start with an even mark, got {group}")
            end = marks[group + 1]
            yield from code[:start]
            yield _Group(group // 2 + 1, list(self._groupify(code[start + 1 : end])))
            yield from self._groupify(code[end + 1 :])
        else:
            yield from code

    @singledispatchmethod
    def _visit(self, op):
        raise NotImplementedError(op)

    @_visit.register
    def _visit_any(self, op: Any):
        return rr.Terminal("<any>")

    @_visit.register
    def _visit_at(self, op: At):
        descriptions = {
            sre_constants.AT_BEGINNING: "start of string",
            sre_constants.AT_BEGINNING_STRING: "beginning of string",
            sre_constants.AT_END: "end of string",
            sre_constants.AT_END_STRING: "end of string",
            sre_constants.AT_UNI_BOUNDARY: "at word boundary",
            sre_constants.AT_UNI_NON_BOUNDARY: "not at word boundary",
        }
        return rr.Terminal(f"<{descriptions[op.where]}>")

    @_visit.register
    def _visit_assert(self, op: Assert):
        return rr.Group(
            self._visit_op_seq(op.pattern), "Must match, but not part of match"
        )

    @_visit.register
    def _visit_assert_not(self, op: AssertNot):
        return rr.Group(self._visit_op_seq(op.pattern), "Must not match")

    @_visit.register
    def _visit_branch(self, op: Branch):
        targets = set()
        branches = []
        optional = False
        for branch in op.branches:
            jump = branch[-1]
            if not isinstance(jump, Jump):
                raise ValueError("Last op in branch not a jump")
            targets.add(jump.pos + jump.skip)
            if len(branch) > 1:
                self._expected_jumps.append(jump)
                branches.append(self._visit_op_seq(branch))
            else:
                optional = True
        # XXX verify that target is actually next op?
        if len(targets) != 1:
            raise ValueError("Not all branches jump to same target")
        if not branches:
            return None
        node = rr.Choice(0, *branches)
        if optional:
            node = rr.Optional(node)
        return node

    @_visit.register
    def _visit_group_ref(self, op: GroupRef):
        return rr.Terminal(f"<group {op.group + 1}>")

    @_visit.register
    def _visit_in(self, op: In):
        negate = "not " if op.negate else ""
        return rr.Terminal(negate + " or ".join(map(self._range_str, op.ranges)))

    @_visit.register
    def _visit_info(self, op: Info):
        pass

    @_visit.register
    def _visit_group(self, op: _Group):
        return rr.Group(self._visit_op_seq(op.ops), f"Group {op.group}")

    @_visit.register
    def _visit_jump(self, op: Jump):
        if (expected := self._expected_jumps.pop()) != op:
            raise ValueError(f"Unexpected jump: {op}, expected {expected}")

    @_visit.register
    def _visit_literal(self, op: Literal):
        return rr.Terminal(self._literal_str(op))

    @_visit.register
    def _lisit_literal_seq(self, op: _LiteralSeq):
        return rr.Terminal("".join(self._literal_str(op) for op in op.literals))

    @_visit.register
    def _visit_not_literal(self, op: NotLiteral):
        return rr.Terminal("not " + self._literal_str(op))

    @_visit.register
    def _visit_repeat(self, op: Repeat):
        return self._visit_repeat_common(op, self._visit_op_seq(op.body))

    @_visit.register
    def _visit_repeat_one(self, op: RepeatOne):
        return self._visit_repeat_common(op, self._visit(op.op))

    def _visit_repeat_common(self, op, body):
        repeat = []
        kind = rr.OneOrMore if op.min_times != 0 else rr.ZeroOrMore
        if op.min_times > 0:
            repeat.append(f"min {op.min_times}")
        if op.max_times != sre_constants.MAXREPEAT:
            repeat.append(f"max {op.max_times}")
        desc = [", ".join(repeat) + " times"] if repeat else []
        if op.minimal:
            desc.append("minimal")
        repeat = rr.Comment(", ".join(desc)) if desc else None
        return kind(body, repeat=repeat)

    @_visit.register
    def _visit_success(self, op: Success):
        # XXX what to do here?
        pass

    def _literal_str(self, literal: Literal):
        char = chr(literal.literal)
        if char == " ":
            return "<space>"
        elif char.isspace():
            return repr(char)[1:-1]
        else:
            return char

    def _range_str(self, op: Op):
        if isinstance(op, Range):
            return f"{chr(op.min_char)} - {chr(op.max_char)}"
        elif isinstance(op, Category):
            return {
                sre_constants.CATEGORY_UNI_SPACE: "unicode space",
                sre_constants.CATEGORY_UNI_NOT_SPACE: "not a unicode space",
            }[op.category]
        elif isinstance(op, Literal):
            return self._literal_str(op)
        raise ValueError(op)


## Main


def plot_re(pattern: str):
    parser = _Parser(_Tokenizer(pattern, 0))
    code = parser.parse()
    return rr.Diagram(_Plotter().plot(code))


if __name__ == "__main__":
    import sys

    plot_re(sys.argv[1]).writeSvg(sys.stdout.write)
