from __future__ import annotations

import sre_constants
import sre_parse
import sre_compile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import count
from typing import DefaultDict, Iterable, List, Optional, Sequence, Tuple


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
class At(Op):
    where: int


@dataclass(eq=False)
class Branch(Op):
    branches: List[List[Op]]


@dataclass(eq=False)
class Failure(Op):
    pass


@dataclass(eq=False)
class In(Op):
    skip: int
    ranges: List[Op]


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
class Range(Op):
    min_char: int
    max_char: int


@dataclass(eq=False)
class Repeat(Op):
    skip: int
    min_times: int
    max_times: int
    body: List[Op]
    epilogue: int


@dataclass(eq=False)
class Success(Op):
    pass


@dataclass(eq=False)
class RepeatOne(Op):
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
            self.at,
            self.branch,
            self.failure,
            self.in_,
            self.info,
            self.jump,
            self.literal,
            self.mark,
            self.range,
            self.repeat,
            self.repeat_one,
            self.success,
        )

    def ops(self) -> Optional[List[Op]]:
        return self._loop(self.op)

    def any(self) -> Optional[Any]:
        if op := self._expect(sre_constants.ANY):
            return Any(op.pos)
        return None

    def at(self):
        if op := self._expect(sre_constants.AT):

            def e(value):
                return lambda: self._expect(value)

            if (
                where := self._alt(
                    e(sre_constants.AT_BEGINNING),
                    e(sre_constants.AT_BEGINNING_LINE),
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

    def failure(self):
        if op := self._expect(sre_constants.FAILURE):
            return Failure(op.pos)

    def in_(self):
        if op := self._expect(sre_constants.IN):
            if (skip := self.int()) is not None:
                with self.tokenizer.limit_to_next(skip - 2):
                    ranges = self.ops()
                if self._expect(sre_constants.FAILURE):
                    return In(op.pos, skip, ranges)

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

    def range(self):
        if op := self._expect(sre_constants.RANGE):
            (min_char, max_char) = self._ints(2)
            if max_char is not None:
                return Range(op.pos, min_char, max_char)

    def repeat(self):
        if op := self._expect(sre_constants.REPEAT):
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
                                op.pos, skip, min_times, max_times, body, epilogue
                            )

    def repeat_one(self):
        if op := self._expect(sre_constants.REPEAT_ONE):
            (skip, min_times, max_times) = self._ints(3)
            if max_times is not None:
                with self.tokenizer.limit_to_next(skip - 1):
                    if (loop_op := self.op()) is not None:
                        if success := self.success():
                            return RepeatOne(
                                op.pos,
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


## Automaton


@dataclass(eq=False)
class State:
    """
    A state in an NFA.
    """

    transitions: List[Tuple[CharacterBag, State]] = field(default_factory=list)
    descr: Optional[str] = None
    _shape: Optional[str] = None

    def add_transition(self, char: CharacterBag, state: State):
        self.transitions.append((char, state))

    def only_epsilon_transition(self):
        return (
            len(self.transitions) == 1
            and CharacterBag.epsilon() == self.transitions[0][0]
        )

    def dump(self) -> Iterable[str]:
        state_number = count()
        state_names: DefaultDict[State, str] = defaultdict(
            lambda: f"p{next(state_number)}"
        )
        yield "digraph {"
        for state in _find_all(self, lambda _: True):
            if state._shape or state.descr:
                attrs = ",".join(
                    f'{k}="{v}"'
                    for (k, v) in {("shape", state._shape), ("label", state.descr)}
                    if v
                )
                yield f"  {state_names[state]} [{attrs}]"
            for (char, new_state) in state.transitions:
                label = char.range_repr()
                yield f'  {state_names[state]} -> {state_names[new_state]} [label="{label}"]'
        yield "}"


@dataclass(eq=False)
class AcceptingState(State):
    _shape = "doublecircle"


@dataclass(eq=False)
class Kleene(State):
    pass


# Some special characters
epsilon = object()
at_end = object()


class CharacterBag:
    @classmethod
    def epsilon(cls):
        return cls([], epsilon=True)

    @classmethod
    def range(cls, start: int, stop: int):
        if start < 0:
            raise ValueError("start < 0")
        elif start > stop:
            raise ValueError("start must be smaller or equal to stop")
        return cls([(start, stop)])

    @classmethod
    def singleton(cls, character):
        if character is epsilon:
            return cls.epsilon()
        else:
            return cls.range(character, character)

    def __init__(self, ranges, epsilon=False):
        self._contains_epsilon = epsilon
        self._ranges = self._merge_ranges(sorted(ranges))

    def min(self):
        return self._ranges[0][0]

    def __bool__(self):
        return self._contains_epsilon or bool(self._ranges)

    def __eq__(self, other):
        if isinstance(other, CharacterBag):
            return (
                self._ranges == other._ranges
                and self._contains_epsilon == other._contains_epsilon
            )
        return NotImplemented

    def range_repr(self) -> str:
        if (
            not self._contains_epsilon
            and len(self._ranges) == 1
            and self.min() == self._ranges[0][1]
        ):
            return chr(self.min())
        elif not self._ranges and self._contains_epsilon:
            return "ε"
        else:
            ranges_repr = ", ".join(
                f"{chr(start)}-{chr(stop)}" for (start, stop) in self._ranges
            )
            epsilon_repr = ", ε" if self._contains_epsilon else ""
            return f"[{ranges_repr}]{epsilon_repr}"

    def __repr__(self):
        return f"<CharacterBag({self.range_repr()})>"

    @staticmethod
    def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(ranges) < 2:
            return ranges

        def merge():
            (prev_start, prev_stop) = ranges[0]
            for (start, stop) in ranges[1:]:
                if start > prev_stop:
                    yield (prev_start, prev_stop)
                    prev_start = start
                prev_stop = stop
            yield (prev_start, prev_stop)

        return list(merge())


def _find_all(entry_state, predicate):
    seen = set()
    to_do = deque([entry_state])
    while to_do:
        state = to_do.pop()
        seen.add(state)
        if predicate(state):
            yield state
        to_do.extend(
            next_state
            for (_, next_state) in state.transitions
            if next_state not in seen
        )


def _remove_epsilon_only(nfa: State):
    """
    Removes all states that are only entered via an ε transition. Changes the given NFA in-place.
    """
    entries = defaultdict(list)
    for state in _find_all(nfa, lambda _: True):
        for (char, next_state) in state.transitions:
            entries[next_state].append((char, state))
    for (state, from_transitions) in entries.items():
        if len(from_transitions) == 1:
            (char, from_state) = from_transitions[0]
            if char == CharacterBag.epsilon():
                i = from_state.transitions.index((char, state))
                from_state.transitions[i : i + 1] = state.transitions


def to_nfa(ops: Sequence[Op]):
    def visit(op: Op, incoming_state: State):
        new_state = None
        if isinstance(op, At):
            # XXX incomplete / not correct
            if op.where != sre_constants.AT_BEGINNING:
                char = {int(sre_constants.AT_END): at_end}[op.where]
                new_state = State()
                incoming_state.add_transition(CharacterBag.singleton(char), new_state)
        elif isinstance(op, Branch):
            branch_entry_state = State(descr="branch")
            new_state = branch_exit_state = State(descr="branch end")
            incoming_state.add_transition(CharacterBag.epsilon(), branch_entry_state)
            for branch in op.branches:
                branch_state = State()
                branch_entry_state.add_transition(CharacterBag.epsilon(), branch_state)
                assert isinstance(branch[-1], Jump)
                (_, branch_last_state) = visit_nodes(branch[:-1], branch_state)
                branch_last_state.add_transition(
                    CharacterBag.epsilon(), branch_exit_state
                )
        elif isinstance(op, Info):
            # Ignore for now
            pass
        elif isinstance(op, Literal):
            new_state = State(descr="literal matched")
            incoming_state.add_transition(CharacterBag.singleton(op.literal), new_state)
        elif isinstance(op, Repeat):
            new_state = Kleene(descr="*")
            incoming_state.add_transition(CharacterBag.epsilon(), new_state)
            (new_state, repeat_end_state) = visit_nodes(op.body, new_state)
            repeat_end_state.add_transition(CharacterBag.epsilon(), new_state)
        elif isinstance(op, Success):
            new_state = AcceptingState()
            incoming_state.add_transition(CharacterBag.epsilon(), new_state)
        else:
            print(f"[WARN] Unhandled op: {type(op).__name__}")
        return new_state if new_state is not None else incoming_state

    def visit_nodes(ops: Sequence[Op], state: State):
        start = state
        for op in ops:
            state = visit(op, state)
        return (start, state)

    nfa = visit_nodes(ops, State())[0]
    _remove_epsilon_only(nfa)
    return nfa


## Main


def plot_re(pattern: str):
    parser = _Parser(_Tokenizer(pattern, 0))
    code = parser.parse()
    nfa = to_nfa(code)
    print("\n".join(nfa.dump()))


if __name__ == "__main__":
    import sys

    plot_re(sys.argv[1])
