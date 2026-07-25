"""
pipeline/rules.py

Rule system for the CPU pipeline scheduler.

Supports two rule formats:

1. Declarative when/then rules (new YAML architecture):
   Used with 'validation_only' configs.  Each rule has a 'when' condition
   block and a 'then' action block parsed from the YAML at load time.

2. Column-constraint rules (original architecture):
   Used with configs that have a 'pipeline.stages' section.  Kept for
   backward compatibility with the old YAML format.

The RuleChecker detects which format each rule uses based on its structure
and instantiates the appropriate class.

pipeline_count has been removed.  Stage capacities are absolute values from
the config.
"""

import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

# Register helpers live in instructions.py to avoid a circular import
# between rules.py and dependency_graph.py.
from .instructions import _get_written_reg_values, _get_read_reg_values

if TYPE_CHECKING:
    from .config import PipelineConfig, StageSequenceGraph
    from .instructions import Instruction

# ---------------------------------------------------------------------------
# Expression DSL
#
# Evaluates YAML condition expressions of the form:
#   writes(producer)
#   stage_index(stage_of(current))
#   resource_used(stage(current), column(current))
#
# These are parsed into a nested tuple AST at rule-load time so that each
# rule evaluation call only needs to traverse the pre-built tree.
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _split_call_args(args_str: str) -> List[str]:
    """
    Split a comma-separated argument string, respecting nested parentheses.

        'a, b, func(c, d)' → ['a', 'b', 'func(c, d)']
    """
    args: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(args_str):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            args.append(args_str[start:i].strip())
            start = i + 1
    tail = args_str[start:].strip()
    if tail:
        args.append(tail)
    return args


def _parse_expr(s: str) -> tuple:
    """
    Parse an expression string into a nested tuple AST.

    Grammar:
        expr      ::= call | literal | variable
        call      ::= IDENT '(' (expr (',' expr)*)? ')'
        literal   ::= INTEGER
        variable  ::= IDENT

    Examples:
        'writes(producer)'
            → ('call', 'writes', [('var', 'producer')])
        'stage_index(stage_of(producer))'
            → ('call', 'stage_index', [('call', 'stage_of', [('var', 'producer')])])
        'resource_used(stage(current), column(current))'
            → ('call', 'resource_used', [
                   ('call', 'stage',  [('var', 'current')]),
                   ('call', 'column', [('var', 'current')])])
        '42'  → ('lit', 42)
        'x'   → ('var', 'x')
    """
    s = s.strip()
    # Function call: name(args)
    paren_open = s.find('(')
    if paren_open > 0 and s.endswith(')'):
        func_name = s[:paren_open].strip()
        if _IDENT_RE.match(func_name):
            args_str    = s[paren_open + 1:-1]
            parsed_args = [
                _parse_expr(a)
                for a in _split_call_args(args_str)
                if a.strip()
            ]
            return ('call', func_name, parsed_args)
    # Integer literal
    try:
        return ('lit', int(s))
    except ValueError:
        pass
    # Variable name (including names with '*' like 'i*')
    return ('var', s)


def _eval_expr(expr: tuple, binding: Dict, ctx: EvalContext) -> Any:
    """
    Evaluate a pre-parsed expression AST tuple against the current
    iteration binding and EvalContext.

    Args:
        expr:    Tuple produced by _parse_expr().
        binding: Current iteration variables, e.g.
                 {'producer': {'row': 0, 'instruction': instr, 'stage': 'X'},
                  'consumer': {'row': 1, 'instruction': instr, 'stage': 'D'}}
        ctx:     Current EvalContext (provides grid, stage graph, capacities).

    Returns:
        A Python value: Set[str] for register sets, int for numeric values,
        str for stage names, None when resolution fails.
    """
    kind = expr[0]
    if kind == 'lit':
        return expr[1]
    if kind == 'var':
        return binding.get(expr[1])
    if kind == 'call':
        evaled = [_eval_expr(a, binding, ctx) for a in expr[2]]
        return _apply_func(expr[1], evaled, ctx)
    return None


def _apply_func(func_name: str, args: List[Any], ctx: EvalContext) -> Any:
    """
    Apply a named DSL function to its already-evaluated arguments.

    Supported functions:
        writes(bound_var)                      Set[str]  written register values
        reads(bound_var)                       Set[str]  read register values
        stage_of(bound_var)                    str       stage name from binding
        stage(bound_var)                       str       alias for stage_of
        stage_index(stage_name)                int       BFS depth in stage graph
                                                          (-1 if unknown or no graph)
        column(bound_var)                      int       col index from binding
        resource_used(stage, col)              int       count of stage in that column
        stage_capacity(stage)                  int       declared capacity
                                                          (999999 when 0 = unbounded)
        when_done(bound_var)                   str       when_done stage name
        when_start(bound_var)                  str       when_start stage name
        column_of_stage(bound_var, stage_name) int|None  col index of first occurrence
    """
    if not args:
        return None

    if func_name == 'writes':
        bound = args[0]
        instr = bound.get('instruction') if isinstance(bound, dict) else None
        return _get_written_reg_values(instr)

    if func_name == 'reads':
        bound = args[0]
        instr = bound.get('instruction') if isinstance(bound, dict) else None
        return _get_read_reg_values(instr)

    if func_name in ('stage_of', 'stage'):
        bound = args[0]
        return bound.get('stage', '') if isinstance(bound, dict) else ''

    if func_name == 'stage_index':
        stage_name = args[0]
        if (ctx.stage_graph is not None
                and isinstance(stage_name, str) and stage_name):
            return ctx.stage_graph.stage_depth.get(stage_name, -1)
        return -1

    if func_name == 'column':
        bound = args[0]
        return bound.get('col', 0) if isinstance(bound, dict) else 0

    if func_name == 'resource_used':
        stage = args[0]
        col   = int(args[1]) if len(args) > 1 and args[1] is not None else 0
        if isinstance(stage, str):
            return sum(
                1 for r in range(ctx.rows)
                if ctx.grid_data.get((r, col)) == stage
            )
        return 0

    if func_name == 'stage_capacity':
        stage = args[0]
        if isinstance(stage, str):
            cap = ctx.stage_capacities.get(stage, 0)
            # Capacity 0 = unbounded: return a value too large to be exceeded
            # so that the gt check in capacity_rule always evaluates to False.
            return cap if cap != 0 else 999999
        return 0

    if func_name == 'when_done':
        bound = args[0]
        instr = bound.get('instruction') if isinstance(bound, dict) else None
        fmt   = getattr(instr, 'format', None) if instr else None
        return getattr(fmt, 'when_done', '') if fmt else ''

    if func_name == 'when_start':
        bound = args[0]
        instr = bound.get('instruction') if isinstance(bound, dict) else None
        fmt   = getattr(instr, 'format', None) if instr else None
        return getattr(fmt, 'when_start', '') if fmt else ''

    if func_name == 'column_of_stage':
        bound      = args[0]
        stage_name = args[1] if len(args) > 1 else None
        if not isinstance(bound, dict) or not isinstance(stage_name, str):
            return None
        row = bound.get('row')
        if row is None:
            return None
        for col in range(ctx.cols):
            if ctx.grid_data.get((row, col)) == stage_name:
                return col
        return None

    # ── Instruction format field accessors ──────────────────────────────────

    if func_name == 'values_committed':
        # values_committed(bound_var) → stage name string
        # Returns the stage where this instruction's result is committed to
        # the architectural register file. Empty string if not declared.
        bound = args[0]
        instr = bound.get('instruction') if isinstance(bound, dict) else None
        fmt = getattr(instr, 'format', None) if instr else None
        return getattr(fmt, 'values_committed', '') if fmt else ''

    if func_name == 'decoded':
        # decoded(bound_var) → stage name string
        # Returns the stage where this instruction is decoded. Used as the
        # earliest valid bypass target. Empty string if not declared.
        bound = args[0]
        instr = bound.get('instruction') if isinstance(bound, dict) else None
        fmt = getattr(instr, 'format', None) if instr else None
        return getattr(fmt, 'decoded', '') if fmt else ''

    if func_name == 'stage_attr':
        # stage_attr(bound_var, 'attribute_name') → any InstructionFormat field
        #
        # General-purpose accessor for any InstructionFormat attribute.
        # Avoids the need to add a dedicated DSL function for every new field.
        #
        # Example (YAML expression):
        #   column_of_stage(producer, stage_attr(producer, 'values_committed'))
        #
        # This lets purely YAML-defined rules read any field added to
        # InstructionFormat without requiring a Python change.
        bound = args[0]
        attr_name = str(args[1]) if len(args) > 1 else None
        if not isinstance(bound, dict) or not attr_name:
            return ''
        instr = bound.get('instruction')
        fmt = getattr(instr, 'format', None) if instr else None
        return getattr(fmt, attr_name, '') if fmt else ''

    warnings.warn(
        f"_apply_func: unknown DSL function '{func_name}'; returning None.",
        UserWarning, stacklevel=2,
    )
    return None

def _extract_eq_exprs_from_when(
        when_block: Optional[Dict],
) -> Optional[Tuple[tuple, tuple]]:
    """
    Scan a 'when' block recursively for the first 'eq' condition and
    return its two pre-parsed expression AST tuples.

    Searches the top level first, then descends into any 'all' list.
    Returns None if no 'eq' condition is found anywhere in the block.

    Used by QuantifiedYAMLRule to determine which register sets to
    display in violation messages without hard-coding RAW/WAR/WAW labels.

    Args:
        when_block: The parsed 'when:' dict from a YAML rule entry,
                    or None.

    Returns:
        Tuple of two pre-parsed AST tuples, or None.
    """
    if not isinstance(when_block, dict):
        return None

    if 'eq' in when_block:
        exprs = when_block['eq']
        if isinstance(exprs, list) and len(exprs) >= 2:
            return (_parse_expr(str(exprs[0])), _parse_expr(str(exprs[1])))

    # Descend into 'all' sub-conditions
    for item in when_block.get('all', []):
        result = _extract_eq_exprs_from_when(item)
        if result is not None:
            return result

    # Descend into 'any' sub-conditions
    for item in when_block.get('any', []):
        result = _extract_eq_exprs_from_when(item)
        if result is not None:
            return result

    return None

# ---------------------------------------------------------------------------
# Violation
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    """Represents a single rule violation found in the current grid state."""
    rule_name:          str
    cells:              List[Tuple[int, int]]
    rows:               List[int]
    message:            str
    annotation_indices: List[int]      = field(default_factory=list)
    highlight_color:    Optional[str]  = None


# ---------------------------------------------------------------------------
# Evaluation context
# ---------------------------------------------------------------------------

@dataclass
class EvalContext:
    """
    All state available to condition evaluators at check time.

    Attributes:
        grid_data:          Maps (row, col) to block type string
        instructions:       Maps row index to Instruction instance
        rows:               Current grid row count
        cols:               Current grid column count
        bypass_annotations: List of BypassAnnotation instances
        binding:            Variable bindings for the current iteration step
        stage_graph:        StageSequenceGraph or None for old-format configs
        stage_capacities:   Maps stage name to absolute column capacity
        dep_graph:          DependencyGraph instance or None.
                             Typed as Any to avoid importing dependency_graph.py
                             here, which would create a circular dependency.
    """
    grid_data:          Dict[Tuple[int, int], str]
    instructions:       Dict[int, Any]
    rows:               int
    cols:               int
    bypass_annotations: List[Any]
    binding:            Dict[str, Any]   = field(default_factory=dict)
    stage_graph:        Optional[Any]    = None
    stage_capacities:   Dict[str, int]   = field(default_factory=dict)
    dep_graph:          Optional[Any]    = None


# ---------------------------------------------------------------------------
# Condition result
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    """Returned by every condition's evaluate() method."""
    matched:             bool                   = False
    annotation_indices:  List[int]              = field(default_factory=list)
    cells:               List[Tuple[int, int]]  = field(default_factory=list)
    rows:                List[int]              = field(default_factory=list)

# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

class AllCondition:
    """
    Logical AND: every sub-condition must match.

    Affected items are the union of all sub-conditions' affected items.
    If any sub-condition does not match, returns matched=False with
    empty affected-item lists.
    """

    def __init__(self, conditions: List):
        self.conditions = conditions

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        combined = ConditionResult(matched=True)
        for cond in self.conditions:
            result = cond.evaluate(ctx)
            if not result.matched:
                return ConditionResult(matched=False)
            combined.annotation_indices.extend(result.annotation_indices)
            combined.cells.extend(result.cells)
            combined.rows.extend(result.rows)
        # Deduplicate while preserving order
        combined.annotation_indices = list(dict.fromkeys(combined.annotation_indices))
        combined.cells              = list(dict.fromkeys(combined.cells))
        combined.rows               = list(dict.fromkeys(combined.rows))
        return combined


class BypassExistsCondition:
    """
    Matches when at least one bypass annotation is present.

    bypass_exists: true  → matches when annotations exist
    bypass_exists: false → matches when no annotations exist
    """

    def __init__(self, value: bool = True):
        self.value = value

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        exists  = len(ctx.bypass_annotations) > 0
        matched = (exists == self.value)
        indices = list(range(len(ctx.bypass_annotations))) if matched else []
        return ConditionResult(matched=matched, annotation_indices=indices)


class DependencyDetectedCondition:
    """
    Matches when at least one bypass annotation has the given type.

    dependency_detected: RAW  → find all annotations labeled 'RAW'

    Case-insensitive comparison against annotation_type.

    In the current 'validation_only' model the user labels annotations
    manually, so this condition checks those labels rather than performing
    semantic analysis of the instruction stream.  The architecture is
    designed to support semantic analysis in a future update.
    """

    def __init__(self, dep_type: str):
        self.dep_type = dep_type.upper()

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        matching = [
            i for i, a in enumerate(ctx.bypass_annotations)
            if a.annotation_type.upper() == self.dep_type
        ]
        return ConditionResult(
            matched=len(matching) > 0,
            annotation_indices=matching,
            rows=sorted({
                row
                for i in matching
                for row in (
                    ctx.bypass_annotations[i].source[0],
                    ctx.bypass_annotations[i].target[0],
                )
            }),
        )


class GridCellOverlappingCondition:
    """
    Matches when the same stage type appears more than once in a column.

    Interprets 'structural overlap' as: two or more rows sharing the
    same stage at the same column index, which indicates a resource
    conflict in the pipeline.

    grid_cell_overlapping: true  → detect column conflicts
    grid_cell_overlapping: false → match only when no conflicts exist
    """

    def __init__(self, value: bool = True):
        self.value = value

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        conflicting_cells: List[Tuple[int, int]] = []

        for col in range(ctx.cols):
            stage_positions: Dict[str, List[Tuple[int, int]]] = {}
            for row in range(ctx.rows):
                cell = (row, col)
                if cell in ctx.grid_data:
                    stage = ctx.grid_data[cell]
                    stage_positions.setdefault(stage, []).append(cell)
            for positions in stage_positions.values():
                if len(positions) > 1:
                    conflicting_cells.extend(positions)

        has_conflict = len(conflicting_cells) > 0
        matched      = (has_conflict == self.value)

        if not matched:
            return ConditionResult(matched=False)

        return ConditionResult(
            matched=True,
            cells=conflicting_cells,
            rows=sorted({c[0] for c in conflicting_cells}),
        )


class InstructionTypeCondition:
    """
    Matches when any row's instruction has a mnemonic in the given list.

    instruction_type: [bne, jr, jal]

    When matched, returns:
    - The row indices where the instruction appears
    - The indices of any bypass annotations whose source or target
      row has such an instruction (so they can be highlighted)

    Designed to flag control hazards by finding branches and jumps
    and highlighting their associated bypass connections.
    """

    def __init__(self, instruction_types: List[str]):
        self.instruction_types: Set[str] = {t.lower() for t in instruction_types}

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        matching_rows = [
            row for row, instr in ctx.instructions.items()
            if instr and instr.name.lower() in self.instruction_types
        ]

        if not matching_rows:
            return ConditionResult(matched=False)

        matching_row_set = set(matching_rows)
        annotation_indices = [
            i for i, a in enumerate(ctx.bypass_annotations)
            if a.source[0] in matching_row_set
            or a.target[0] in matching_row_set
        ]

        return ConditionResult(
            matched=True,
            rows=sorted(matching_rows),
            annotation_indices=annotation_indices,
        )


class NeverCondition:
    """
    Fallback for unrecognised condition types.  Never matches.
    Allows configs with unknown conditions to load without crashing.
    """

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        return ConditionResult(matched=False)

# ---------------------------------------------------------------------------
# YAML-driven conditions
# ---------------------------------------------------------------------------

class AnyCondition:
    """
    Logical OR: matched when at least one sub-condition matches.

    Used in YAML as:
        any:
          - gt: [...]
          - not:
              in_pipeline_order: ...
    """

    def __init__(self, conditions: List):
        self.conditions = conditions

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        for cond in self.conditions:
            result = cond.evaluate(ctx)
            if result.matched:
                return result
        return ConditionResult(matched=False)


class NotCondition:
    """
    Logical NOT: inverts the matched state of its sub-condition.

    Affected items (annotation_indices, cells, rows) from the sub-condition
    are preserved so the frontend can still highlight related elements.

    Used in YAML as:
        not:
          bypass_exists:
            args: [producer, consumer]
    """

    def __init__(self, condition):
        self.condition = condition

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        result = self.condition.evaluate(ctx)
        return ConditionResult(
            matched=not result.matched,
            annotation_indices=result.annotation_indices,
            cells=result.cells,
            rows=result.rows,
        )


class EqCondition:
    """
    Matched when two expressions are equal, or — when both evaluate to sets —
    when their intersection is non-empty (overlap semantics used for
    register-hazard detection).

    Used in YAML as:
        eq:
          - writes(producer)
          - reads(consumer)
    """

    def __init__(self, expr1_str: str, expr2_str: str):
        self._e1 = _parse_expr(expr1_str)
        self._e2 = _parse_expr(expr2_str)

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        v1 = _eval_expr(self._e1, ctx.binding, ctx)
        v2 = _eval_expr(self._e2, ctx.binding, ctx)
        if isinstance(v1, set) and isinstance(v2, set):
            return ConditionResult(matched=bool(v1 & v2))
        return ConditionResult(matched=(v1 == v2))


class LtCondition:
    """
    Matched when the numeric value of expr1 is strictly less than expr2.

    Returns matched=False (rather than raising) when either side cannot be
    cast to int — e.g. when stage_index returns -1 for an unknown stage.

    Used in YAML as:
        lt:
          - stage_index(stage_of(producer))
          - stage_index(stage_of(consumer))
    """

    def __init__(self, expr1_str: str, expr2_str: str):
        self._e1 = _parse_expr(expr1_str)
        self._e2 = _parse_expr(expr2_str)

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        v1 = _eval_expr(self._e1, ctx.binding, ctx)
        v2 = _eval_expr(self._e2, ctx.binding, ctx)
        try:
            return ConditionResult(matched=(int(v1) < int(v2)))
        except (TypeError, ValueError):
            return ConditionResult(matched=False)


class GtCondition:
    """
    Matched when the numeric value of expr1 is strictly greater than expr2.

    Used in YAML as:
        gt:
          - resource_used(stage(current), column(current))
          - stage_capacity(stage(current))
    """

    def __init__(self, expr1_str: str, expr2_str: str):
        self._e1 = _parse_expr(expr1_str)
        self._e2 = _parse_expr(expr2_str)

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        v1 = _eval_expr(self._e1, ctx.binding, ctx)
        v2 = _eval_expr(self._e2, ctx.binding, ctx)
        try:
            return ConditionResult(matched=(int(v1) > int(v2)))
        except (TypeError, ValueError):
            return ConditionResult(matched=False)


class InPipelineOrderCondition:
    """
    Matched when stage2 IS reachable from stage1 in the pipeline graph, or
    they are the same stage.

    This condition represents the VALID state.  Wrap it in NotCondition to
    detect violations:

        not:
          in_pipeline_order:
            args:
              - stage_of(previous)
              - stage_of(current)

    Returns matched=True (valid) when the stage graph is unavailable so
    unknown stages fail safe rather than generating false violations.

    stage_graph is taken from ctx.stage_graph at evaluation time so the
    condition does not need a reference to the config at build time.
    """

    def __init__(self, expr1_str: str, expr2_str: str):
        self._e1 = _parse_expr(expr1_str)
        self._e2 = _parse_expr(expr2_str)

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        stage1 = _eval_expr(self._e1, ctx.binding, ctx)
        stage2 = _eval_expr(self._e2, ctx.binding, ctx)

        if not isinstance(stage1, str) or not isinstance(stage2, str):
            return ConditionResult(matched=True)  # Unknown → assume valid

        graph = ctx.stage_graph
        if graph is None:
            return ConditionResult(matched=True)  # No graph → assume valid

        in_order = (stage1 == stage2) or graph.is_reachable(stage1, stage2)
        return ConditionResult(matched=in_order)


class BypassExistsBetweenCondition:
    """
    Matched when any bypass annotation runs from producer_var's row to
    consumer_var's row.

    Distinct from the existing BypassExistsCondition (which checks whether
    ANY annotations exist at all).  This condition checks for a specific
    directed connection between two named binding variables.

    Used in YAML as:
        bypass_exists:
          args: [producer, consumer]

    Returns the matched annotation indices so that NotCondition can surface
    them for highlighting in the violation output.
    """

    def __init__(self, producer_var: str, consumer_var: str):
        self._producer_var = producer_var
        self._consumer_var = consumer_var

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        producer = ctx.binding.get(self._producer_var, {})
        consumer = ctx.binding.get(self._consumer_var, {})

        if not isinstance(producer, dict) or not isinstance(consumer, dict):
            return ConditionResult(matched=False)

        prod_row = producer.get('row')
        cons_row = consumer.get('row')
        if prod_row is None or cons_row is None:
            return ConditionResult(matched=False)

        matching = [
            i for i, a in enumerate(ctx.bypass_annotations)
            if a.source[0] == prod_row and a.target[0] == cons_row
        ]

        return ConditionResult(
            matched=len(matching) > 0,
            annotation_indices=matching,
        )


class TemporalHazardCondition:
    """
    Configurable temporal hazard condition.

    Compares the column index of one stage on the producer's row to the
    column index of one stage on the consumer's row.  Which stages are
    compared is determined by InstructionFormat attribute names supplied
    in the YAML, making this condition reusable for RAW, WAW, and WAR
    without any Python changes.

    Returns matched=False when either referenced stage is absent from the
    current grid layout so partial rows never generate false positives.

    YAML syntax:
        - temporal_hazard:
            args: [producer_var, consumer_var]
            producer_stage_attr: values_committed   # default
            consumer_stage_attr: when_start         # default
            operator: gt                            # default

    Supported operators:
        gt  — col(producer_stage) >  col(consumer_stage)
        gte — col(producer_stage) >= col(consumer_stage)
        lt  — col(producer_stage) <  col(consumer_stage)
        lte — col(producer_stage) <= col(consumer_stage)

    RAW example: producer_stage_attr=values_committed, consumer_stage_attr=when_start, operator=gt
        Fires when the producer commits to the register file (C) strictly
        AFTER the consumer begins execution (X), meaning the consumer
        would read stale data without a bypass.

    WAW example: producer_stage_attr=values_committed, consumer_stage_attr=values_committed, operator=gt
        Fires when the consumer commits before the producer, which could
        leave the wrong value in the register file.
    """

    _OPS = {
        'gt':  lambda a, b: a > b,
        'gte': lambda a, b: a >= b,
        'lt':  lambda a, b: a < b,
        'lte': lambda a, b: a <= b,
    }

    def __init__(self,
                 producer_var:         str,
                 consumer_var:         str,
                 producer_stage_attr:  str = 'values_committed',
                 consumer_stage_attr:  str = 'when_start',
                 operator:             str = 'gt'):
        self._producer_var        = producer_var
        self._consumer_var        = consumer_var
        self._producer_stage_attr = producer_stage_attr
        self._consumer_stage_attr = consumer_stage_attr
        self._op = self._OPS.get(operator, self._OPS['gt'])

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        producer = ctx.binding.get(self._producer_var, {})
        consumer = ctx.binding.get(self._consumer_var, {})

        if not isinstance(producer, dict) or not isinstance(consumer, dict):
            return ConditionResult(matched=False)

        prod_row   = producer.get('row')
        cons_row   = consumer.get('row')
        prod_instr = producer.get('instruction')
        cons_instr = consumer.get('instruction')

        if prod_row is None or cons_row is None:
            return ConditionResult(matched=False)

        prod_fmt = getattr(prod_instr, 'format', None) if prod_instr else None
        cons_fmt = getattr(cons_instr, 'format', None) if cons_instr else None

        # Look up which stage name to search for from the instruction format
        prod_stage_name = getattr(prod_fmt, self._producer_stage_attr, '') if prod_fmt else ''
        cons_stage_name = getattr(cons_fmt, self._consumer_stage_attr, '') if cons_fmt else ''

        # Cannot evaluate if either stage is not declared on this instruction
        if not prod_stage_name or not cons_stage_name:
            return ConditionResult(matched=False)

        # Find the column containing that stage in each row
        prod_col: Optional[int] = next(
            (c for c in range(ctx.cols)
             if ctx.grid_data.get((prod_row, c)) == prod_stage_name),
            None,
        )
        cons_col: Optional[int] = next(
            (c for c in range(ctx.cols)
             if ctx.grid_data.get((cons_row, c)) == cons_stage_name),
            None,
        )

        # Cannot evaluate if either stage is not yet placed in the grid
        if prod_col is None or cons_col is None:
            return ConditionResult(matched=False)

        return ConditionResult(matched=self._op(prod_col, cons_col))


class TemporalBypassExistsCondition:
    """
    Configurable temporal bypass existence check.

    A bypass annotation is considered valid when ALL of the following hold:

        source.row == producer_var.row
        source.col >= col(source_stage_attr of producer row)

        target.row == consumer_var.row
        target.col >= col(target_min_stage_attr of consumer row)
        target.col <  col(target_max_stage_attr of consumer row)

    Any bound that references a stage not yet placed in the grid is
    relaxed (the constraint is skipped for that bound), so partial
    layouts do not incorrectly clear violations.

    YAML syntax:
        - temporal_bypass_exists:
            args: [producer_var, consumer_var]
            source_stage_attr:     when_done    # default
            target_min_stage_attr: decoded      # default
            target_max_stage_attr: when_start   # default

    For RAW (add instruction with when_done=X, decoded=D, when_start=X):
        source.col >= col(X of producer row)      — bypass from X or later
        col(D of consumer row) <= target.col      — not before decode
        target.col <  col(X of consumer row)      — before execution starts

    This admits exactly the four documented bypass paths:
        X → D,  X → I,  W → D,  W → I
    """

    def __init__(self,
                 producer_var:          str,
                 consumer_var:          str,
                 source_stage_attr:     str = 'when_done',
                 target_min_stage_attr: str = 'decoded',
                 target_max_stage_attr: str = 'when_start'):
        self._producer_var          = producer_var
        self._consumer_var          = consumer_var
        self._source_stage_attr     = source_stage_attr
        self._target_min_stage_attr = target_min_stage_attr
        self._target_max_stage_attr = target_max_stage_attr

    def _find_col(self, row: int, stage: str, ctx: EvalContext) -> Optional[int]:
        """Return the first column index containing *stage* in *row*, or None."""
        if not stage:
            return None
        return next(
            (c for c in range(ctx.cols)
             if ctx.grid_data.get((row, c)) == stage),
            None,
        )

    def evaluate(self, ctx: EvalContext) -> ConditionResult:
        producer = ctx.binding.get(self._producer_var, {})
        consumer = ctx.binding.get(self._consumer_var, {})

        if not isinstance(producer, dict) or not isinstance(consumer, dict):
            return ConditionResult(matched=False)

        prod_row   = producer.get('row')
        cons_row   = consumer.get('row')
        prod_instr = producer.get('instruction')
        cons_instr = consumer.get('instruction')

        if prod_row is None or cons_row is None:
            return ConditionResult(matched=False)

        prod_fmt = getattr(prod_instr, 'format', None) if prod_instr else None
        cons_fmt = getattr(cons_instr, 'format', None) if cons_instr else None

        # Resolve stage names from instruction format attributes
        source_stage     = getattr(prod_fmt, self._source_stage_attr,     '') if prod_fmt else ''
        target_min_stage = getattr(cons_fmt, self._target_min_stage_attr, '') if cons_fmt else ''
        target_max_stage = getattr(cons_fmt, self._target_max_stage_attr, '') if cons_fmt else ''

        # Resolve column bounds (None = stage absent → relax that bound)
        source_min_col  = self._find_col(prod_row, source_stage,     ctx)
        target_min_col  = self._find_col(cons_row, target_min_stage, ctx)
        target_max_col  = self._find_col(cons_row, target_max_stage, ctx)

        matching: List[int] = []
        for i, a in enumerate(ctx.bypass_annotations):

            # Must connect the right rows
            if a.source[0] != prod_row or a.target[0] != cons_row:
                continue

            # Source must be at or after when_done of the producer
            if source_min_col is not None and a.source[1] < source_min_col:
                continue  # source placed before when_done; skip

            # Target must be at or after decoded of the consumer
            if target_min_col is not None and a.target[1] < target_min_col:
                continue  # target placed before decode; skip

            # Target must be strictly before when_start of the consumer
            if target_max_col is not None and a.target[1] >= target_max_col:
                continue  # target at or after execution start; skip

            matching.append(i)

        return ConditionResult(
            matched=len(matching) > 0,
            annotation_indices=matching,
        )

# ---------------------------------------------------------------------------
# Condition factory  (replaces the previous build_condition)
# ---------------------------------------------------------------------------

def build_condition(when_block: Dict):
    """
    Parse a 'when' dict from a YAML rule entry and return a condition object.

    Supported keys:
        all                    → AllCondition
        any                    → AnyCondition
        not                    → NotCondition
        eq                     → EqCondition
        lt                     → LtCondition
        gt                     → GtCondition
        in_pipeline_order      → InPipelineOrderCondition
        temporal_hazard        → TemporalHazardCondition
        temporal_bypass_exists → TemporalBypassExistsCondition
        bypass_exists          → BypassExistsBetweenCondition (args form)
                                  BypassExistsCondition       (bool form)
        dependency_detected    → DependencyDetectedCondition
        grid_cell_overlapping  → GridCellOverlappingCondition
        instruction_type       → InstructionTypeCondition

    Unknown keys produce a UserWarning and return NeverCondition.
    """
    if 'all' in when_block:
        subs = [
            build_condition(item)
            for item in when_block['all']
            if isinstance(item, dict)
        ]
        return AllCondition(subs)

    if 'any' in when_block:
        subs = [
            build_condition(item)
            for item in when_block['any']
            if isinstance(item, dict)
        ]
        return AnyCondition(subs)

    if 'not' in when_block:
        inner = when_block['not']
        if isinstance(inner, dict):
            return NotCondition(build_condition(inner))
        warnings.warn(
            f"build_condition: 'not' value must be a mapping; "
            f"got {type(inner).__name__}; rule will never fire.",
            UserWarning, stacklevel=2,
        )
        return NeverCondition()

    if 'eq' in when_block:
        exprs = when_block['eq']
        if isinstance(exprs, list) and len(exprs) >= 2:
            return EqCondition(str(exprs[0]), str(exprs[1]))
        warnings.warn(
            "build_condition: 'eq' requires a list of two expression strings.",
            UserWarning, stacklevel=2,
        )
        return NeverCondition()

    if 'lt' in when_block:
        exprs = when_block['lt']
        if isinstance(exprs, list) and len(exprs) >= 2:
            return LtCondition(str(exprs[0]), str(exprs[1]))
        warnings.warn(
            "build_condition: 'lt' requires a list of two expression strings.",
            UserWarning, stacklevel=2,
        )
        return NeverCondition()

    if 'gt' in when_block:
        exprs = when_block['gt']
        if isinstance(exprs, list) and len(exprs) >= 2:
            return GtCondition(str(exprs[0]), str(exprs[1]))
        warnings.warn(
            "build_condition: 'gt' requires a list of two expression strings.",
            UserWarning, stacklevel=2,
        )
        return NeverCondition()

    if 'in_pipeline_order' in when_block:
        val = when_block['in_pipeline_order']
        if isinstance(val, dict):
            args = val.get('args', [])
            if isinstance(args, list) and len(args) >= 2:
                return InPipelineOrderCondition(str(args[0]), str(args[1]))
        warnings.warn(
            "build_condition: 'in_pipeline_order' requires {args: [expr1, expr2]}.",
            UserWarning, stacklevel=2,
        )
        return NeverCondition()

    if 'temporal_hazard' in when_block:
        val = when_block['temporal_hazard']
        if isinstance(val, dict):
            args = val.get('args', [])
            if isinstance(args, list) and len(args) >= 2:
                return TemporalHazardCondition(
                    producer_var=str(args[0]),
                    consumer_var=str(args[1]),
                    producer_stage_attr=val.get(
                        'producer_stage_attr', 'values_committed'
                    ),
                    consumer_stage_attr=val.get(
                        'consumer_stage_attr', 'when_start'
                    ),
                    operator=val.get('operator', 'gt'),
                )
        warnings.warn(
            "build_condition: 'temporal_hazard' requires "
            "{args: [producer_var, consumer_var]} plus optional "
            "producer_stage_attr, consumer_stage_attr, operator.",
            UserWarning, stacklevel=2,
        )
        return NeverCondition()

    if 'temporal_bypass_exists' in when_block:
        val = when_block['temporal_bypass_exists']
        if isinstance(val, dict):
            args = val.get('args', [])
            if isinstance(args, list) and len(args) >= 2:
                return TemporalBypassExistsCondition(
                    producer_var=str(args[0]),
                    consumer_var=str(args[1]),
                    source_stage_attr=val.get(
                        'source_stage_attr', 'when_done'
                    ),
                    target_min_stage_attr=val.get(
                        'target_min_stage_attr', 'decoded'
                    ),
                    target_max_stage_attr=val.get(
                        'target_max_stage_attr', 'when_start'
                    ),
                )
        warnings.warn(
            "build_condition: 'temporal_bypass_exists' requires "
            "{args: [producer_var, consumer_var]} plus optional "
            "source_stage_attr, target_min_stage_attr, target_max_stage_attr.",
            UserWarning, stacklevel=2,
        )
        return NeverCondition()

    if 'bypass_exists' in when_block:
        val = when_block['bypass_exists']
        if isinstance(val, dict) and 'args' in val:
            args = val['args']
            if isinstance(args, list) and len(args) >= 2:
                return BypassExistsBetweenCondition(str(args[0]), str(args[1]))
        return BypassExistsCondition(bool(val))

    if 'dependency_detected' in when_block:
        return DependencyDetectedCondition(str(when_block['dependency_detected']))

    if 'grid_cell_overlapping' in when_block:
        return GridCellOverlappingCondition(bool(when_block['grid_cell_overlapping']))

    if 'instruction_type' in when_block:
        types = when_block['instruction_type']
        if isinstance(types, str):
            types = [types]
        return InstructionTypeCondition(list(types))

    warnings.warn(
        f"build_condition: unrecognised condition keys {list(when_block.keys())}; "
        f"rule will never fire.",
        UserWarning, stacklevel=2,
    )
    return NeverCondition()


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class HighlightAction:
    """
    Produces a Violation that instructs the frontend to highlight specific
    elements.

    target == 'bypass_annotation': highlight SVG arrow elements by index
    target == 'grid_cell':         highlight grid cell divs by coordinate

    The color is looked up from bypass_type_colors at build time so the
    action carries the resolved hex string rather than a symbolic reference.
    """

    def __init__(self, target: str, color: str):
        """
        Args:
            target: 'bypass_annotation' or 'grid_cell'
            color:  Resolved hex color string for the highlight
        """
        self.target = target
        self.color  = color

    def execute(self,
                rule_name: str,
                result:    ConditionResult) -> List[Violation]:
        """
        Convert a ConditionResult into a list of Violation objects.

        Returns a single Violation containing all affected items, or an
        empty list if the result has no affected items to highlight.

        Args:
            rule_name: Name of the parent rule (used in Violation.rule_name)
            result:    ConditionResult from the condition evaluation

        Returns:
            List[Violation]: Zero or one Violation.
        """
        if self.target == 'bypass_annotation':
            if not result.annotation_indices:
                return []
            return [Violation(
                rule_name=rule_name,
                cells=[],
                rows=result.rows,
                message=rule_name,
                annotation_indices=result.annotation_indices,
                highlight_color=self.color,
            )]

        if self.target == 'grid_cell':
            if not result.cells:
                return []
            return [Violation(
                rule_name=rule_name,
                cells=result.cells,
                rows=result.rows,
                message=rule_name,
                annotation_indices=[],
                highlight_color=self.color,
            )]

        return []


# ---------------------------------------------------------------------------
# Action factory
# ---------------------------------------------------------------------------

def build_action(then_block: Dict,
                 bypass_type_colors: Dict[str, str]) -> HighlightAction:
    """
    Parse a 'then' dict from a YAML rule and return the appropriate
    action object.

    Currently only the 'highlight' action type is supported.
    color_ref must match a key in bypass_type_colors; falls back to
    '#cccccc' (gray) if not found.

    Args:
        then_block:         The parsed 'then:' dict from a YAML rule entry
        bypass_type_colors: Maps bypass type name to hex color string;
                            from {bt.name: bt.color for bt in config.to_bypass_types()}

    Returns:
        HighlightAction instance.
    """
    action    = then_block.get('action', {})
    target    = action.get('target', 'grid_cell')
    color_ref = action.get('color_ref', '')
    color     = bypass_type_colors.get(color_ref, '#cccccc')
    return HighlightAction(target=target, color=color)


# ---------------------------------------------------------------------------
# Abstract Rule base
# ---------------------------------------------------------------------------

class Rule(ABC):
    """
    Abstract base class for all pipeline scheduling rules.

    Both WhenThenRule (new YAML format) and ColumnConstraintRule (old
    format) inherit from this class.  RuleChecker calls check() on all
    enabled rules and aggregates their violations.
    """

    def __init__(self, name: str, description: str, enabled: bool = True):
        """
        Args:
            name:        Rule name; used as the key in set_rule_enabled()
            description: Description shown in the rules panel
            enabled:     False means the rule is skipped in check_all()
        """
        self.name:        str  = name
        self.description: str  = description
        self.enabled:     bool = enabled

    @abstractmethod
    def check(self,
              grid_data:          Dict[Tuple[int, int], str],
              instructions:       Dict[int, 'Instruction'],
              rows:               int,
              cols:               int,
              bypass_annotations: List[Any] = None,
              dep_graph:          Any       = None) -> List[Violation]:
        """
        Evaluate the rule against the current scheduler state.

        Args:
            grid_data:          Maps (row, col) to block type string
            instructions:       Maps row index to Instruction instance
            rows:               Current grid row count
            cols:               Current grid column count
            bypass_annotations: List of BypassAnnotation instances
            dep_graph:          DependencyGraph instance or None.
                                 QuantifiedYAMLRule uses this for dep-class
                                 iteration; other rule types ignore it.
        """

# ---------------------------------------------------------------------------
# Helpers for QuantifiedYAMLRule
# ---------------------------------------------------------------------------

# Default highlight colours when none is specified in the YAML then block
_QUANTIFIED_DEFAULT_COLORS: Dict[str, str] = {
    'dependency_rule': '#cc0000',
    'ordering_rule':   '#006600',
    'capacity_rule':   '#0066cc',
}


class _TrueCondition:
    """
    Always-matched sentinel used when a quantified rule has no 'when' block.

    In that case every iteration binding triggers a violation, which is
    useful during rule authoring to confirm the iteration is working
    before adding conditions.
    """
    def evaluate(self, ctx: EvalContext) -> ConditionResult:  # noqa: N802
        return ConditionResult(matched=True)


# ---------------------------------------------------------------------------
# QuantifiedYAMLRule
# ---------------------------------------------------------------------------

class QuantifiedYAMLRule(Rule):
    """
    A single Rule class covering dependency_rule, ordering_rule, and
    capacity_rule without hard-coded subclasses.

    When dep_class is set (e.g. 'RAW', 'WAW', 'WAR'), dependency_rule
    iteration is driven by the DependencyGraph rather than an all-pairs
    sweep, so only instruction pairs with actual register conflicts are
    evaluated.  The YAML 'when' block then only needs to express temporal
    constraints; register matching is handled by the graph.

    When dep_graph is None or dep_class is empty, falls back to an all-pairs
    sweep for backward compatibility.
    """

    def __init__(
            self,
            name:             str,
            description:      str,
            enabled:          bool,
            condition,
            color:            str,
            label:            str,
            rule_type:        str,
            stage_graph,
            stage_capacities: Dict[str, int],
            when_block:       Optional[Dict] = None,
            dep_class:        str            = '',
    ):
        super().__init__(name, description, enabled)
        self._condition        = condition
        self._color            = color
        self._label            = label
        self._rule_type        = rule_type
        self._stage_graph      = stage_graph
        self._stage_capacities = stage_capacities
        self._dep_class        = dep_class

        # Pre-parse the first eq pair in the when block as a fallback for
        # _build_message when shared_regs is not available from a dep_graph edge.

        self._eq_exprs: Optional[Tuple[tuple, tuple]] = (
            _extract_eq_exprs_from_when(when_block)
            if isinstance(when_block, dict) else None
        )

    def update_stage_capacities(self, stage_capacities: Dict[str, int]) -> None:
        self._stage_capacities = stage_capacities

    # ── check() ──────────────────────────────────────────────────────────────

    def check(
            self,
            grid_data:          Dict[Tuple[int, int], str],
            instructions:       Dict[int, Any],
            rows:               int,
            cols:               int,
            bypass_annotations: List[Any] = None,
            dep_graph:          Any       = None,
    ) -> List[Violation]:
        bypass_annotations = bypass_annotations or []
        violations: List[Violation] = []

        for binding, affected in self._iterate(
                grid_data, instructions, rows, cols, dep_graph
        ):
            ctx = EvalContext(
                grid_data=grid_data,
                instructions=instructions,
                rows=rows,
                cols=cols,
                bypass_annotations=bypass_annotations,
                binding=binding,
                stage_graph=self._stage_graph,
                stage_capacities=self._stage_capacities,
                dep_graph=dep_graph,
            )
            result = self._condition.evaluate(ctx)
            if not result.matched:
                continue

            merged_cells = list(dict.fromkeys(
                affected.get('cells', []) + result.cells
            ))
            merged_rows = sorted(set(
                affected.get('rows', []) + result.rows
            ))
            merged_ann = list(dict.fromkeys(
                affected.get('annotation_indices', []) + result.annotation_indices
            ))

            violations.append(Violation(
                rule_name=self.name,
                cells=merged_cells,
                rows=merged_rows,
                message=self._build_message(binding),
                annotation_indices=merged_ann,
                highlight_color=self._color,
            ))

        return violations

    # ── Iteration dispatch ───────────────────────────────────────────────────

    def _iterate(
            self,
            grid_data:    Dict[Tuple[int, int], str],
            instructions: Dict[int, Any],
            rows:         int,
            cols:         int,
            dep_graph:    Any = None,
    ):
        if self._rule_type == 'dependency_rule':
            yield from self._iter_instruction_pairs(
                grid_data, instructions, rows, cols, dep_graph
            )
        elif self._rule_type == 'ordering_rule':
            yield from self._iter_consecutive_stages(grid_data, rows, cols)
        elif self._rule_type == 'capacity_rule':
            yield from self._iter_column_stages(grid_data, rows, cols)

    # ── Iteration strategies ─────────────────────────────────────────────────

    @staticmethod
    def _last_stage(row: int,
                    grid_data: Dict[Tuple[int, int], str],
                    cols: int) -> str:
        """Return the stage name of the rightmost occupied cell in a row."""
        for col in range(cols - 1, -1, -1):
            if (row, col) in grid_data:
                return grid_data[(row, col)]
        return ''

    def _iter_instruction_pairs(
            self,
            grid_data:    Dict[Tuple[int, int], str],
            instructions: Dict[int, Any],
            rows:         int,
            cols:         int,
            dep_graph:    Any = None,
    ):
        """
        Yield (binding, affected) for each (producer, consumer) pair.

        When dep_graph is provided and dep_class is set, iterates only the
        edges from the dependency graph that match dep_class — so the YAML
        'when' block only needs to express temporal constraints (register
        matching is already encoded in the graph).

        Falls back to all-pairs iteration when dep_graph is absent or
        dep_class is empty, for backward compatibility with rules that
        express register matching via eq conditions.
        """
        if dep_graph is not None and self._dep_class:
            for edge in dep_graph.get_edges_by_class(self._dep_class):
                prod_row  = edge.producer_row
                cons_row  = edge.consumer_row
                prod_instr = instructions.get(prod_row)
                cons_instr = instructions.get(cons_row)
                if not prod_instr or not cons_instr:
                    continue

                prod_cells = [
                    (prod_row, c) for c in range(cols)
                    if (prod_row, c) in grid_data
                ]
                cons_cells = [
                    (cons_row, c) for c in range(cols)
                    if (cons_row, c) in grid_data
                ]

                binding = {
                    'producer': {
                        'row':         prod_row,
                        'instruction': prod_instr,
                        'stage':       self._last_stage(prod_row, grid_data, cols),
                        # shared_regs from the dep graph edge drives _build_message
                        'shared_regs': sorted(edge.shared_regs),
                    },
                    'consumer': {
                        'row':         cons_row,
                        'instruction': cons_instr,
                        'stage':       self._last_stage(cons_row, grid_data, cols),
                    },
                }
                affected = {
                    'cells':              prod_cells + cons_cells,
                    'rows':               [prod_row, cons_row],
                    'annotation_indices': [],
                }
                yield binding, affected

        else:
            # All-pairs fallback
            active_rows = sorted(
                r for r, instr in instructions.items()
                if instr and getattr(instr, 'name', '')
            )
            for p_idx, prod_row in enumerate(active_rows):
                prod_instr = instructions[prod_row]
                prod_cells = [
                    (prod_row, c) for c in range(cols)
                    if (prod_row, c) in grid_data
                ]
                for cons_row in active_rows[p_idx + 1:]:
                    cons_cells = [
                        (cons_row, c) for c in range(cols)
                        if (cons_row, c) in grid_data
                    ]
                    binding = {
                        'producer': {
                            'row':         prod_row,
                            'instruction': prod_instr,
                            'stage':       self._last_stage(prod_row, grid_data, cols),
                            'shared_regs': [],
                        },
                        'consumer': {
                            'row':         cons_row,
                            'instruction': instructions[cons_row],
                            'stage':       self._last_stage(cons_row, grid_data, cols),
                        },
                    }
                    affected = {
                        'cells':              prod_cells + cons_cells,
                        'rows':               [prod_row, cons_row],
                        'annotation_indices': [],
                    }
                    yield binding, affected

    def _iter_consecutive_stages(
            self,
            grid_data: Dict[Tuple[int, int], str],
            rows:      int,
            cols:      int,
    ):
        for row in range(rows):
            row_stages = sorted(
                [(col, grid_data[(row, col)])
                 for col in range(cols) if (row, col) in grid_data],
                key=lambda x: x[0],
            )
            for i in range(len(row_stages) - 1):
                col1, stage1 = row_stages[i]
                col2, stage2 = row_stages[i + 1]
                binding = {
                    'previous': {'row': row, 'col': col1, 'stage': stage1},
                    'current':  {'row': row, 'col': col2, 'stage': stage2},
                }
                affected = {
                    'cells': [(row, col1), (row, col2)],
                    'rows':  [row],
                    'annotation_indices': [],
                }
                yield binding, affected

    def _iter_column_stages(
            self,
            grid_data: Dict[Tuple[int, int], str],
            rows:      int,
            cols:      int,
    ):
        for col in range(cols):
            by_stage: Dict[str, List[Tuple[int, int]]] = {}
            for row in range(rows):
                if (row, col) in grid_data:
                    stage = grid_data[(row, col)]
                    by_stage.setdefault(stage, []).append((row, col))
            for stage, positions in by_stage.items():
                binding = {
                    'current': {
                        'stage':     stage,
                        'col':       col,
                        'positions': positions,
                        'count':     len(positions),
                    },
                }
                affected = {
                    'cells': positions,
                    'rows':  sorted({p[0] for p in positions}),
                    'annotation_indices': [],
                }
                yield binding, affected

    # ── Message builder ───────────────────────────────────────────────────────

    def _build_message(self, binding: Dict) -> str:
        label = self._label or self.name

        if self._rule_type == 'dependency_rule':
            prod       = binding.get('producer', {})
            cons       = binding.get('consumer', {})
            prod_name  = getattr(prod.get('instruction'), 'name', '?')
            cons_name  = getattr(cons.get('instruction'), 'name', '?')

            # Prefer shared_regs from the dep graph edge (populated by
            # _iter_instruction_pairs when dep_graph is available).
            shared = prod.get('shared_regs', [])

            # Fallback: re-evaluate the first eq pair from the when block.
            if not shared and self._eq_exprs is not None:
                _ctx = EvalContext(
                    grid_data={}, instructions={}, rows=0, cols=0,
                    bypass_annotations=[], binding=binding,
                )
                v1 = _eval_expr(self._eq_exprs[0], binding, _ctx)
                v2 = _eval_expr(self._eq_exprs[1], binding, _ctx)
                if isinstance(v1, set) and isinstance(v2, set):
                    shared = sorted(v1 & v2)

            shared_str = f" {list(shared)}" if shared else ''
            return (
                f"{label}: row {prod.get('row', '?')} ({prod_name}){shared_str}"
                f" → row {cons.get('row', '?')} ({cons_name}) — no bypass"
            )

        if self._rule_type == 'ordering_rule':
            prev = binding.get('previous', {})
            curr = binding.get('current', {})
            return (
                f"{label}: row {curr.get('row', '?')} —"
                f" '{curr.get('stage', '?')}' at col {curr.get('col', '?')}"
                f" does not follow '{prev.get('stage', '?')}'"
                f" at col {prev.get('col', '?')} in pipeline order"
            )

        if self._rule_type == 'capacity_rule':
            curr = binding.get('current', {})
            cap  = self._stage_capacities.get(curr.get('stage', ''), '?')
            return (
                f"{label}: '{curr.get('stage', '?')}'"
                f" used {curr.get('count', '?')}×"
                f" in col {curr.get('col', '?')}"
                f" (capacity = {cap})"
            )

        return label

# ---------------------------------------------------------------------------
# RuleChecker
# ---------------------------------------------------------------------------

class RuleChecker:
    """
    Manages and evaluates all registered pipeline rules.

    Detects the rule format from the YAML structure:
    - Rules with 'when' and 'then' keys → WhenThenRule
    - Rules with 'type: column_constraint' → ColumnConstraintRule
    - Unknown formats → warning + skip
    """

    def __init__(self):
        self.rules: List[Rule] = []

    def load_rules_from_config(self, config: 'PipelineConfig') -> None:
        """
        Clear existing rules and register rules from the config.

        Dispatch order:
          1. type in {dependency_rule, ordering_rule, capacity_rule}
             → QuantifiedYAMLRule
          2. 'when' + 'then' (no type or unknown type)
             → WhenThenRule
          3. type == column_constraint
             → ColumnConstraintRule (legacy)
          4. anything else → UserWarning + skip
        """
        self.rules.clear()

        bypass_type_colors = {
            bt.name: bt.color for bt in config.to_bypass_types()
        }
        stage_capacities = config.to_stage_capacities()
        stage_graph = config.to_stage_sequence_graph()

        _QUANTIFIED_TYPES = frozenset({
            'dependency_rule', 'ordering_rule', 'capacity_rule',
        })

        for rule_def in config.to_rule_definitions():
            rule_name = rule_def.get('name', '<unnamed>')
            description = str(rule_def.get('description', '')).strip()
            enabled = bool(rule_def.get('enabled', True))
            rule_type = rule_def.get('type', '')
            dep_class = str(rule_def.get('dep_class', ''))

            if rule_type in _QUANTIFIED_TYPES:
                then_block = rule_def.get('then', {})
                highlight = (
                    then_block.get('highlight', {})
                    if isinstance(then_block, dict) else {}
                )
                color = (
                        highlight.get('color')
                        or bypass_type_colors.get(highlight.get('color_ref', ''), '')
                        or _QUANTIFIED_DEFAULT_COLORS.get(rule_type, '#cccccc')
                )
                label = highlight.get('label', rule_name)
                when_block = rule_def.get('when')
                condition = (
                    build_condition(when_block)
                    if isinstance(when_block, dict)
                    else _TrueCondition()
                )
                self.rules.append(QuantifiedYAMLRule(
                    name=rule_name,
                    description=description,
                    enabled=enabled,
                    condition=condition,
                    color=color,
                    label=label,
                    rule_type=rule_type,
                    stage_graph=stage_graph,
                    stage_capacities=stage_capacities,
                    when_block=when_block,
                    dep_class=dep_class,
                ))
                continue

            warnings.warn(
                f"RuleChecker: rule '{rule_name}' has no recognised format "
                f"(type='{rule_type}', dep_class='{dep_class}'); skipping.",
                UserWarning, stacklevel=2,
            )

    def add_rule(self, rule: Rule) -> None:
        """
        Manually append a rule.  Cleared on the next load_rules_from_config call.

        Args:
            rule: Any Rule subclass instance.
        """
        self.rules.append(rule)

    def get_rules_info(self) -> List[Dict]:
        """
        Return metadata for all registered rules.

        Returns:
            List[Dict]: Each dict contains name, description, enabled.
        """
        return [
            {
                'name':        rule.name,
                'description': rule.description,
                'enabled':     rule.enabled,
            }
            for rule in self.rules
        ]

    def set_rule_enabled(self, rule_name: str, enabled: bool) -> bool:
        """
        Enable or disable a rule by name.

        Returns:
            bool: True if found and updated; False if not found.
        """
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = enabled
                return True
        return False

    def set_all_enabled(self, enabled: bool) -> None:
        """Enable or disable all rules."""
        for rule in self.rules:
            rule.enabled = enabled


    def check_all(self,
                  grid_data: Dict[Tuple[int, int], str],
                  instructions: Dict[int, 'Instruction'],
                  rows: int,
                  cols: int,
                  bypass_annotations: List[Any] = None,
                  dep_graph: Any = None) -> List[Violation]:
        """
        Run all enabled rules and aggregate their violations.

        Args:
            dep_graph: DependencyGraph instance, rebuilt by the scheduler
                       before each call.  Passed to every rule; only
                       QuantifiedYAMLRule with a dep_class uses it.
        """
        bypass_annotations = bypass_annotations or []
        all_violations: List[Violation] = []
        for rule in self.rules:
            if not rule.enabled:
                continue
            all_violations.extend(
                rule.check(
                    grid_data, instructions, rows, cols,
                    bypass_annotations, dep_graph,
                )
            )
        return all_violations