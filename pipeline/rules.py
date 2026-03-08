# pipeline/rules.py
from dataclasses import dataclass
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod

@dataclass
class Violation:
    """
    Represents a pipeline scheduling rule violation.

    Attributes:
        rule_name (str): Name of the violated rule
        cells (List[Tuple[int, int]]): Grid cells involved in violation
        rows (List[int]): Row numbers affected by violation
        message (str): Human-readable violation description
    """
    rule_name: str
    cells: List[Tuple[int, int]]  # List of (row, col) tuples
    rows: List[int]  # List of row numbers for instruction violations
    message: str

class Rule(ABC):
    """
    Abstract base class for pipeline scheduling rules.

    All concrete rule implementations must inherit from this class
    and implement the check() method.
    """

    def __init__(self, name: str, description: str, enabled: bool = True):
        """
        Initialize a rule instance.

        Args:
            name: Human-readable rule name
            description: Detailed description of the rule
            enabled: Whether the rule is active for checking
        """
        self.name = name
        self.description = description
        self.enabled = enabled

    @abstractmethod
    def check(self, grid_data: Dict[Tuple[int, int], str],
              instructions: Dict[int, 'Instruction'],
              rows: int, cols: int, pipeline_count: int) -> List[Violation]:
        """
        Check if the rule is violated in the current pipeline schedule.

        Args:
            grid_data: Dictionary mapping (row, col) tuples to block types
            instructions: Dictionary mapping row numbers to instructions
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            pipeline_count: Number of concurrent pipelines (1 or 2)

        Returns:
            List[Violation]: List of detected violations, empty if none
        """
        pass

class UniqueBlockPerColumnRule(Rule):
    """
    Rule enforcing unique execution blocks per column.

    Ensures that X, Y0, Y1, Y2, and Y3 blocks cannot appear more than
    once in the same column, preventing resource conflicts.
    """

    def __init__(self, enabled: bool = True):
        super().__init__(
            "Unique Execution Blocks Per Column",
            "X, Y0, Y1, Y2, and Y3 blocks cannot occupy the same column more than once",
            enabled=enabled
        )
        self.restricted_blocks = {'X', 'Y0', 'Y1', 'Y2', 'Y3'}

    def check(self, grid_data: Dict[Tuple[int, int], str],
              instructions: Dict[int, 'Instruction'],
              rows: int, cols: int, pipeline_count: int) -> List[Violation]:
        violations = []

        # Check each column
        for col in range(cols):
            # Track which restricted blocks appear in this column
            block_positions: Dict[str, List[Tuple[int, int]]] = {}

            for row in range(rows):
                cell = (row, col)
                if cell in grid_data:
                    block_type = grid_data[cell]
                    if block_type in self.restricted_blocks:
                        if block_type not in block_positions:
                            block_positions[block_type] = []
                        block_positions[block_type].append(cell)

            # Check for duplicates
            for block_type, positions in block_positions.items():
                if len(positions) > 1:
                    affected_rows = sorted(set(pos[0] for pos in positions))
                    violations.append(Violation(
                        rule_name=self.name,
                        cells=positions,
                        rows=affected_rows,
                        message=f"Block '{block_type}' appears {len(positions)} times in column {col}"
                    ))

        return violations

class PipelineStageCountPerColumnRule(Rule):
    """
    Rule enforcing pipeline stage count limits per column.

    Ensures F, D, I, W, and C stages appear at most once per pipeline
    in each column, respecting the number of available pipelines.
    """
    def __init__(self, enabled: bool = True):
        super().__init__(
            "Pipeline Stage Count Per Column",
            "F, D, I, W, and C blocks may appear in a column at most once per pipeline",
            enabled=enabled
        )
        self.stage_blocks = {'F', 'D', 'I', 'W', 'C'}

    def check(self, grid_data: Dict[Tuple[int, int], str],
              instructions: Dict[int, 'Instruction'],
              rows: int, cols: int, pipeline_count: int) -> List[Violation]:
        violations = []

        for col in range(cols):
            # count occurrences of each stage in column
            stage_positions: Dict[str, List[Tuple[int, int]]] = {}
            for row in range(rows):
                cell = (row, col)
                if cell in grid_data:
                    b = grid_data[cell]
                    if b in self.stage_blocks:
                        if b not in stage_positions:
                            stage_positions[b] = []
                        stage_positions[b].append(cell)

            for stage, positions in stage_positions.items():
                if len(positions) > pipeline_count:
                    affected_rows = sorted(set(pos[0] for pos in positions))
                    violations.append(Violation(
                        rule_name=self.name,
                        cells=positions,
                        rows=affected_rows,
                        message=f"Stage '{stage}' appears {len(positions)} times in column {col} (max {pipeline_count})"
                    ))
        return violations

class RuleChecker:
    """
    Manages and executes all pipeline scheduling rules.

    Coordinates rule checking, maintains rule state, and provides
    an interface for enabling/disabling rules.
    """

    def __init__(self):
        self.rules: List[Rule] = []
        self._register_default_rules()

    def _register_default_rules(self):
        """Register the default set of pipeline rules."""
        self.add_rule(UniqueBlockPerColumnRule(enabled=True))
        self.add_rule(PipelineStageCountPerColumnRule(enabled=True))

    def add_rule(self, rule: Rule):
        """
        Add a rule to the checker.

        Args:
            rule: Rule instance to add
        """
        self.rules.append(rule)

    def check_all(self, grid_data: Dict[Tuple[int, int], str],
                  instructions: Dict[int, 'Instruction'],
                  rows: int, cols: int,
                  pipeline_count: int) -> List[Violation]:
        """
        Check all enabled rules and return violations.

        Args:
            grid_data: Dictionary mapping (row, col) tuples to block types
            instructions: Dictionary mapping row numbers to instructions
            rows: Number of rows in the grid
            cols: Number of columns in the grid
            pipeline_count: Number of concurrent pipelines

        Returns:
            List[Violation]: All detected violations across all enabled rules
        """
        all_violations = []
        for rule in self.rules:
            if not getattr(rule, 'enabled', True):
                continue
            violations = rule.check(grid_data, instructions, rows, cols, pipeline_count)
            all_violations.extend(violations)
        return all_violations

    def get_rules_info(self) -> List[Dict]:
        """
        Get information about all registered rules.

        Returns:
            List[Dict]: List of dictionaries containing rule name, description, and enabled status
        """
        return [
            {
                'name': rule.name,
                'description': rule.description,
                'enabled': getattr(rule, 'enabled', True)
            }
            for rule in self.rules
        ]

    def set_rule_enabled(self, rule_name: str, enabled: bool) -> bool:
        """
        Enable or disable a specific rule by name.

        Args:
            rule_name: Name of the rule to modify
            enabled: Whether to enable (True) or disable (False) the rule

        Returns:
            bool: True if rule was found and updated, False otherwise
        """
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = enabled
                return True
        return False

    def set_all_enabled(self, enabled: bool):
        """
        Enable or disable all rules.

        Args:
            enabled: Whether to enable (True) or disable (False) all rules
        """
        for rule in self.rules:
            rule.enabled = enabled