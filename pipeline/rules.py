"""
pipeline/rules.py

Defines the Rule abstract base class, Violation dataclass,
ColumnConstraintRule, and the RuleChecker that manages all rules.

Rules are no longer hard-coded. RuleChecker.load_rules_from_config()
reads rule definitions from a PipelineConfig and instantiates the
appropriate Rule subclass for each entry based on its 'type' field.

The 'column_constraint' type is the only type currently supported.
Unknown types produce a UserWarning and are skipped so that a config
file with one unknown rule type still loads its other rules correctly.

pipeline_count has been removed from all method signatures. Stage
capacity limits are absolute values read directly from the config.
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only during static analysis to avoid circular imports at
    # runtime (config.py imports nothing from the pipeline package).
    from .config import PipelineConfig
    from .instructions import Instruction


# ---------------------------------------------------------------------------
# Violation
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    """
    Represents a single rule violation detected in the current grid state.

    Attributes:
        rule_name (str):                  Name of the rule that was violated
        cells     (List[Tuple[int,int]]): (row, col) coordinates involved
        rows      (List[int]):            Row numbers whose labels should be
                                          highlighted in the UI
        message   (str):                  Human-readable description built
                                          from the YAML violation_message
                                          template
    """
    rule_name: str
    cells:     List[Tuple[int, int]]
    rows:      List[int]
    message:   str


# ---------------------------------------------------------------------------
# Rule (abstract base)
# ---------------------------------------------------------------------------

class Rule(ABC):
    """
    Abstract base class for all pipeline scheduling rules.

    Concrete subclasses must implement check().  Instances are created
    by RuleChecker.load_rules_from_config() based on the 'type' field
    in each YAML rule definition, or manually via RuleChecker.add_rule()
    for programmatic use.
    """

    def __init__(self, name: str, description: str, enabled: bool = True):
        """
        Initialise a rule.

        Args:
            name:        Human-readable rule name; used as the key in
                         set_rule_enabled() lookups
            description: Detailed description displayed in the rules panel
            enabled:     If False the rule is skipped in check_all()
        """
        self.name:        str  = name
        self.description: str  = description
        self.enabled:     bool = enabled

    @abstractmethod
    def check(self,
              grid_data:    Dict[Tuple[int, int], str],
              instructions: Dict[int, 'Instruction'],
              rows:         int,
              cols:         int) -> List[Violation]:
        """
        Evaluate the rule against the current grid state.

        Args:
            grid_data:    Maps (row, col) tuples to block type strings
            instructions: Maps row numbers to Instruction instances
            rows:         Current number of grid rows
            cols:         Current number of grid columns

        Returns:
            List[Violation]: All violations found; empty list if none
        """


# ---------------------------------------------------------------------------
# ColumnConstraintRule
# ---------------------------------------------------------------------------

class ColumnConstraintRule(Rule):
    """
    Generic rule that enforces per-column stage capacity limits defined
    in the YAML config.

    Interprets a 'logic' dict from the config at runtime.  The expected
    YAML structure for the logic block is:

        logic:
          scope: "column"
          target: "grid_data"
          constraint:
            group_by: "stage"
            limit:
              from: "pipeline.stages"
              key:  "stage"
            ignore_if:
              capacity_equals: 0
            violation_message: >
              Stage '{stage}' exceeds its allowed capacity ({capacity})
              in column {col}.

    Fields used at runtime:
        constraint.ignore_if.capacity_equals
            Stages whose capacity in stage_capacities equals this value
            are exempt from checking.  Set to 0 in the default config
            to mark unbounded stages (e.g. 'i', 'r').
        constraint.violation_message
            Template string; {stage}, {col}, {capacity} are substituted.

    Fields present for documentation / future extensibility but not
    currently interpreted:
        scope, target, constraint.group_by,
        constraint.limit.from, constraint.limit.key
    """

    def __init__(self,
                 name:             str,
                 description:      str,
                 enabled:          bool,
                 logic:            Dict,
                 stage_capacities: Dict[str, int]):
        """
        Create a ColumnConstraintRule.

        Args:
            name:             Rule name (from YAML 'name' field)
            description:      Rule description (from YAML 'description')
            enabled:          Whether the rule is active (from YAML 'enabled')
            logic:            The parsed 'logic' block from the YAML rule entry
            stage_capacities: Maps stage name to absolute column capacity;
                              from PipelineConfig.to_stage_capacities().
                              A value of 0 marks the stage as unbounded.
        """
        super().__init__(name, description, enabled)
        self._logic:            Dict          = logic
        self._stage_capacities: Dict[str, int] = stage_capacities

    def update_stage_capacities(self, stage_capacities: Dict[str, int]) -> None:
        """
        Replace the stage capacity mapping without recreating the rule.

        Called by RuleChecker.load_rules_from_config() when a new config
        is loaded mid-session so existing rule instances are updated in
        place rather than discarded and re-created.

        Args:
            stage_capacities: New mapping from PipelineConfig.to_stage_capacities()
        """
        self._stage_capacities = stage_capacities

    def check(self,
              grid_data:    Dict[Tuple[int, int], str],
              instructions: Dict[int, 'Instruction'],
              rows:         int,
              cols:         int) -> List[Violation]:
        """
        Scan every column for stages that exceed their declared capacity.

        For each column, groups all placed blocks by stage name, looks up
        each stage's capacity in self._stage_capacities, and reports a
        Violation when the block count in that column exceeds the limit.

        Stages with a capacity equal to ignore_if.capacity_equals (typically 0)
        are skipped entirely, making them unbounded.

        Stages present in grid_data but absent from stage_capacities are
        also skipped silently so that unknown block types do not cause errors.

        Args:
            grid_data:    Maps (row, col) to block type string
            instructions: Not used by this rule; present to satisfy the ABC
            rows:         Number of rows to evaluate
            cols:         Number of columns to evaluate

        Returns:
            List[Violation]: One entry per (stage, column) pair that exceeds
                             its capacity limit
        """
        constraint      = self._logic.get('constraint', {})
        ignore_if       = constraint.get('ignore_if', {})
        ignore_capacity = ignore_if.get('capacity_equals')

        # Strip leading/trailing whitespace from YAML block scalars
        violation_template: str = str(constraint.get(
            'violation_message',
            "Stage '{stage}' exceeds its allowed capacity ({capacity}) "
            "in column {col}."
        )).strip()

        violations: List[Violation] = []

        for col in range(cols):
            # Group all blocks in this column by stage name
            stage_positions: Dict[str, List[Tuple[int, int]]] = {}
            for row in range(rows):
                cell = (row, col)
                if cell in grid_data:
                    block_type = grid_data[cell]
                    stage_positions.setdefault(block_type, []).append(cell)

            for stage, positions in stage_positions.items():
                capacity: Optional[int] = self._stage_capacities.get(stage)

                # Stage not declared in config — skip silently
                if capacity is None:
                    continue

                # Honour the ignore_if.capacity_equals exemption from YAML
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

    Rules are loaded from a PipelineConfig via load_rules_from_config()
    rather than being registered at construction time.  A freshly
    constructed RuleChecker has no rules until that method is called.

    New rule types can be added to _RULE_TYPE_REGISTRY without modifying
    load_rules_from_config(); the method dispatches based on the dict.
    """

    # Maps YAML 'type' strings to the Rule subclass that handles them.
    # Extend this dict when new rule types are introduced.
    _RULE_TYPE_REGISTRY: Dict[str, type] = {
        'column_constraint': ColumnConstraintRule,
    }

    def __init__(self):
        self.rules: List[Rule] = []

    # ------------------------------------------------------------------
    # Config-driven rule loading
    # ------------------------------------------------------------------

    def load_rules_from_config(self, config: 'PipelineConfig') -> None:
        """
        Clear existing rules and register rules defined in the config.

        Iterates over config.to_rule_definitions() and instantiates the
        Rule subclass registered under the 'type' field of each entry.

        If the 'type' value is not present in _RULE_TYPE_REGISTRY, a
        UserWarning is issued and the rule is skipped so that a config with
        one unsupported rule type still loads all other rules correctly.

        Args:
            config: A validated PipelineConfig instance whose
                    to_rule_definitions() and to_stage_capacities() methods
                    are used to build rules.
        """
        self.rules.clear()
        stage_capacities = config.to_stage_capacities()

        for rule_def in config.to_rule_definitions():
            rule_type = rule_def.get('type', '')
            rule_name = rule_def.get('name', '<unnamed>')

            if rule_type not in self._RULE_TYPE_REGISTRY:
                warnings.warn(
                    f"RuleChecker: unknown rule type '{rule_type}' for rule "
                    f"'{rule_name}'; skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            rule_class = self._RULE_TYPE_REGISTRY[rule_type]

            if rule_class is ColumnConstraintRule:
                rule: Rule = ColumnConstraintRule(
                    name=rule_name,
                    description=str(rule_def.get('description', '')).strip(),
                    enabled=bool(rule_def.get('enabled', True)),
                    logic=rule_def.get('logic', {}),
                    stage_capacities=stage_capacities,
                )
            else:
                # Generic path for future Rule subclasses that only need
                # the three standard base constructor arguments.
                rule = rule_class(
                    name=rule_name,
                    description=str(rule_def.get('description', '')).strip(),
                    enabled=bool(rule_def.get('enabled', True)),
                )

            self.rules.append(rule)

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: Rule) -> None:
        """
        Manually append a rule to the checker.

        Intended for testing or for adding rules programmatically that
        are not derived from a config file.  Rules added this way are
        cleared the next time load_rules_from_config() is called.

        Args:
            rule: Any Rule subclass instance
        """
        self.rules.append(rule)

    def get_rules_info(self) -> List[Dict]:
        """
        Return metadata for all registered rules.

        Returns:
            List[Dict]: One dict per rule containing:
                'name'        (str):  Rule name
                'description' (str):  Rule description
                'enabled'     (bool): Current enabled state
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

        Args:
            rule_name: Exact rule name string as it was registered
            enabled:   True to enable, False to disable

        Returns:
            bool: True if a matching rule was found and updated,
                  False if no rule with that name exists
        """
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = enabled
                return True
        return False

    def set_all_enabled(self, enabled: bool) -> None:
        """
        Enable or disable all registered rules at once.

        Args:
            enabled: True to enable all rules, False to disable all
        """
        for rule in self.rules:
            rule.enabled = enabled

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def check_all(self,
                  grid_data:    Dict[Tuple[int, int], str],
                  instructions: Dict[int, 'Instruction'],
                  rows:         int,
                  cols:         int) -> List[Violation]:
        """
        Run all enabled rules and aggregate their violations.

        The pipeline_count parameter from the previous version has been
        removed.  Stage capacities are now absolute values supplied
        directly by the config rather than being scaled at check time.

        Args:
            grid_data:    Maps (row, col) tuples to block type strings
            instructions: Maps row numbers to Instruction instances
            rows:         Current grid row count
            cols:         Current grid column count

        Returns:
            List[Violation]: All violations from all enabled rules,
                             in the order rules were registered
        """
        all_violations: List[Violation] = []
        for rule in self.rules:
            if not rule.enabled:
                continue
            all_violations.extend(
                rule.check(grid_data, instructions, rows, cols)
            )
        return all_violations