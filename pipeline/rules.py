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

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import PipelineConfig
    from .instructions import Instruction


# ---------------------------------------------------------------------------
# Violation
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    """
    Represents a single rule violation found in the current grid state.

    Attributes:
        rule_name           (str):                Name of the violated rule
        cells               (List[Tuple[int,int]]): Grid cells to highlight
        rows                (List[int]):           Row label indices to pulse
        message             (str):                 Human-readable description
        annotation_indices  (List[int]):           Indices into the
                                                   bypass_annotations list
                                                   whose SVG elements should
                                                   be highlighted
        highlight_color     (Optional[str]):       Hex color for annotation
                                                   highlights; None means use
                                                   the annotation's own color
    """
    rule_name:          str
    cells:              List[Tuple[int, int]]
    rows:               List[int]
    message:            str
    annotation_indices: List[int]            = field(default_factory=list)
    highlight_color:    Optional[str]        = None


# ---------------------------------------------------------------------------
# Evaluation context
# ---------------------------------------------------------------------------

@dataclass
class EvalContext:
    """
    All state available to condition evaluators at check time.

    Passed by value to every condition's evaluate() call so conditions
    are stateless and do not mutate shared state.

    Attributes:
        grid_data:          Maps (row, col) to block type string
        instructions:       Maps row index to Instruction instance
        rows:               Current grid row count
        cols:               Current grid column count
        bypass_annotations: List of BypassAnnotation instances
    """
    grid_data:          Dict[Tuple[int, int], str]
    instructions:       Dict[int, Any]
    rows:               int
    cols:               int
    bypass_annotations: List[Any]


# ---------------------------------------------------------------------------
# Condition result
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    """
    Returned by every condition's evaluate() method.

    Attributes:
        matched:             True if the condition fired
        annotation_indices:  Indices into EvalContext.bypass_annotations
                             that caused the match
        cells:               Grid cell coordinates involved in the match
        rows:                Row indices involved in the match
    """
    matched:             bool                     = False
    annotation_indices:  List[int]                = field(default_factory=list)
    cells:               List[Tuple[int, int]]    = field(default_factory=list)
    rows:                List[int]                = field(default_factory=list)


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
# Condition factory
# ---------------------------------------------------------------------------

def build_condition(when_block: Dict):
    """
    Parse a 'when' dict from a YAML rule and return the appropriate
    condition object.

    Supported keys:
        all:                  list of sub-condition dicts → AllCondition
        bypass_exists:        bool → BypassExistsCondition
        dependency_detected:  str  → DependencyDetectedCondition
        grid_cell_overlapping: bool → GridCellOverlappingCondition
        instruction_type:     str or list → InstructionTypeCondition

    Unknown keys produce a UserWarning and return NeverCondition.

    Args:
        when_block: The parsed 'when:' dict from a YAML rule entry.

    Returns:
        A condition object with an evaluate(EvalContext) -> ConditionResult
        method.
    """
    if 'all' in when_block:
        sub_conditions = []
        for item in when_block['all']:
            if isinstance(item, dict):
                sub_conditions.append(build_condition(item))
        return AllCondition(sub_conditions)

    if 'bypass_exists' in when_block:
        return BypassExistsCondition(bool(when_block['bypass_exists']))

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
        UserWarning,
        stacklevel=2,
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
              bypass_annotations: List[Any] = None) -> List[Violation]:
        """
        Evaluate the rule against the current scheduler state.

        Args:
            grid_data:          Maps (row, col) to block type string
            instructions:       Maps row index to Instruction instance
            rows:               Current grid row count
            cols:               Current grid column count
            bypass_annotations: List of BypassAnnotation instances;
                                 None is treated as empty list

        Returns:
            List[Violation]: All violations found; empty list if none.
        """


# ---------------------------------------------------------------------------
# WhenThenRule
# ---------------------------------------------------------------------------

class WhenThenRule(Rule):
    """
    A rule defined by a declarative 'when' condition and 'then' action
    parsed from the new YAML format.

    The condition is built by build_condition() and the action by
    build_action() at config-load time.  check() evaluates the condition
    against the current EvalContext and, if matched, executes the action
    to produce Violation objects.
    """

    def __init__(self,
                 name:      str,
                 description: str,
                 enabled:   bool,
                 condition,
                 action:    HighlightAction):
        """
        Args:
            name:        Rule name
            description: Rule description
            enabled:     Initial enabled state
            condition:   Any object with evaluate(EvalContext) -> ConditionResult
            action:      HighlightAction instance
        """
        super().__init__(name, description, enabled)
        self.condition = condition
        self.action    = action

    def check(self,
              grid_data:          Dict[Tuple[int, int], str],
              instructions:       Dict[int, 'Instruction'],
              rows:               int,
              cols:               int,
              bypass_annotations: List[Any] = None) -> List[Violation]:
        """Evaluate condition; if matched, execute action."""
        bypass_annotations = bypass_annotations or []
        ctx    = EvalContext(grid_data, instructions, rows, cols, bypass_annotations)
        result = self.condition.evaluate(ctx)
        if not result.matched:
            return []
        return self.action.execute(self.name, result)


# ---------------------------------------------------------------------------
# ColumnConstraintRule (retained for backward compatibility)
# ---------------------------------------------------------------------------

class ColumnConstraintRule(Rule):
    """
    Retained for backward compatibility with configs that include a
    'pipeline.stages' section and use 'type: column_constraint' rules.

    Enforces per-column stage capacity limits.  Stages with capacity 0
    are unbounded (exempt from checking).

    See the original rules.py for full documentation of this class.
    """

    def __init__(self,
                 name:             str,
                 description:      str,
                 enabled:          bool,
                 logic:            Dict,
                 stage_capacities: Dict[str, int]):
        super().__init__(name, description, enabled)
        self._logic:            Dict           = logic
        self._stage_capacities: Dict[str, int] = stage_capacities

    def update_stage_capacities(self, stage_capacities: Dict[str, int]) -> None:
        self._stage_capacities = stage_capacities

    def check(self,
              grid_data:          Dict[Tuple[int, int], str],
              instructions:       Dict[int, 'Instruction'],
              rows:               int,
              cols:               int,
              bypass_annotations: List[Any] = None) -> List[Violation]:

        constraint      = self._logic.get('constraint', {})
        ignore_if       = constraint.get('ignore_if', {})
        ignore_capacity = ignore_if.get('capacity_equals')
        violation_template = str(constraint.get(
            'violation_message',
            "Stage '{stage}' exceeds its allowed capacity ({capacity}) "
            "in column {col}."
        )).strip()

        violations: List[Violation] = []

        for col in range(cols):
            stage_positions: Dict[str, List[Tuple[int, int]]] = {}
            for row in range(rows):
                cell = (row, col)
                if cell in grid_data:
                    block_type = grid_data[cell]
                    stage_positions.setdefault(block_type, []).append(cell)

            for stage, positions in stage_positions.items():
                capacity: Optional[int] = self._stage_capacities.get(stage)
                if capacity is None:
                    continue
                if ignore_capacity is not None and capacity == ignore_capacity:
                    continue
                if len(positions) > capacity:
                    affected_rows = sorted({pos[0] for pos in positions})
                    message = (violation_template
                               .replace('{stage}',    stage)
                               .replace('{col}',      str(col))
                               .replace('{capacity}', str(capacity)))
                    violations.append(Violation(
                        rule_name=self.name,
                        cells=positions,
                        rows=affected_rows,
                        message=message,
                    ))

        return violations


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

        Builds bypass_type_colors from the config's bypass types so that
        WhenThenRule actions can resolve color_ref names to hex strings
        without needing a reference back to the config at check time.

        Args:
            config: A validated PipelineConfig instance.
        """
        self.rules.clear()

        bypass_type_colors = {
            bt.name: bt.color
            for bt in config.to_bypass_types()
        }
        stage_capacities = config.to_stage_capacities()

        for rule_def in config.to_rule_definitions():
            rule_name = rule_def.get('name', '<unnamed>')
            description = str(rule_def.get('description', '')).strip()
            enabled     = bool(rule_def.get('enabled', True))

            # ── New declarative format ──────────────────────────────
            if 'when' in rule_def and 'then' in rule_def:
                condition = build_condition(rule_def['when'])
                action    = build_action(rule_def['then'], bypass_type_colors)
                self.rules.append(WhenThenRule(
                    name=rule_name,
                    description=description,
                    enabled=enabled,
                    condition=condition,
                    action=action,
                ))
                continue

            # ── Legacy column_constraint format ─────────────────────
            rule_type = rule_def.get('type', '')
            if rule_type == 'column_constraint':
                self.rules.append(ColumnConstraintRule(
                    name=rule_name,
                    description=description,
                    enabled=enabled,
                    logic=rule_def.get('logic', {}),
                    stage_capacities=stage_capacities,
                ))
                continue

            # ── Unknown format ──────────────────────────────────────
            warnings.warn(
                f"RuleChecker: rule '{rule_name}' has no recognised format "
                f"(expected 'when'+'then' or 'type: column_constraint'); "
                f"skipping.",
                UserWarning,
                stacklevel=2,
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
                  grid_data:          Dict[Tuple[int, int], str],
                  instructions:       Dict[int, 'Instruction'],
                  rows:               int,
                  cols:               int,
                  bypass_annotations: List[Any] = None) -> List[Violation]:
        """
        Run all enabled rules and aggregate their violations.

        Args:
            grid_data:          Maps (row, col) to block type string
            instructions:       Maps row index to Instruction instance
            rows:               Current grid row count
            cols:               Current grid column count
            bypass_annotations: List of BypassAnnotation instances;
                                 defaults to empty list

        Returns:
            List[Violation]: All violations from all enabled rules.
        """
        bypass_annotations = bypass_annotations or []
        all_violations: List[Violation] = []
        for rule in self.rules:
            if not rule.enabled:
                continue
            all_violations.extend(
                rule.check(grid_data, instructions, rows, cols, bypass_annotations)
            )
        return all_violations