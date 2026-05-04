"""
pipeline/scheduler.py

Contains PipelineScheduler, the central controller for the CPU instruction
pipeline scheduler, and BypassAnnotation, which represents a single drawn
bypass/hazard annotation connecting two grid cells.

BypassType (formerly PipelineType, formerly defined here) has been moved to
config.py to avoid circular imports.

pipeline_count and all related UI controls have been removed.  Stage column
capacity limits are now defined per-stage in the YAML config file and are
absolute limits with no multiplier.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config       import PipelineConfig, BypassType
from .instructions import Instruction, set_instruction_formats
from .rules        import RuleChecker


# ---------------------------------------------------------------------------
# BypassAnnotation
# ---------------------------------------------------------------------------

@dataclass
class BypassAnnotation:
    """
    Represents a single drawn bypass/hazard annotation connecting two grid cells.

    Instances are created by PipelineScheduler.add_bypass_annotation() and
    stored in PipelineScheduler.bypass_annotations.  They are serialized into
    the saved state JSON under the 'bypass_annotations' key.

    Attributes:
        annotation_type (str):          Matches a BypassType name (e.g. 'RAW')
        color           (str):          Hex colour string inherited from the
                                        BypassType at creation time
        source          (Tuple[int,int]): (row, col) of the circled origin cell
        target          (Tuple[int,int]): (row, col) of the arrowhead destination
    """
    annotation_type: str
    color:           str
    source:          Tuple[int, int]
    target:          Tuple[int, int]

    def to_dict(self) -> Dict:
        """
        Serialize to a JSON-compatible dict.

        Returns:
            dict: Keys are annotation_type, color, source {row, col},
                  target {row, col}.
        """
        return {
            'annotation_type': self.annotation_type,
            'color':           self.color,
            'source': {'row': self.source[0], 'col': self.source[1]},
            'target': {'row': self.target[0], 'col': self.target[1]},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BypassAnnotation':
        """
        Deserialize from a JSON-compatible dict.

        Args:
            data: Dict with the same structure as to_dict() output.

        Returns:
            BypassAnnotation: New instance.
        """
        return cls(
            annotation_type=data['annotation_type'],
            color=data['color'],
            source=(int(data['source']['row']), int(data['source']['col'])),
            target=(int(data['target']['row']), int(data['target']['col'])),
        )


# ---------------------------------------------------------------------------
# PipelineScheduler
# ---------------------------------------------------------------------------

class PipelineScheduler:
    """
    Central controller for the CPU instruction pipeline scheduler.

    Manages the grid state, instructions, bypass annotations, and rule
    checking.  All configuration data (instruction formats, bypass types,
    rule definitions, stage capacities) flows from a PipelineConfig instance
    loaded at construction time or via load_config().

    A single global instance is created in app.py; all Flask routes delegate
    to its methods rather than containing scheduling logic directly.

    pipeline_count has been removed.  Stage column capacities are now
    absolute values defined per-stage in the YAML config file.
    """

    def __init__(self, config: PipelineConfig = None):
        """
        Create a new scheduler.

        Loads configuration from the hard-coded CONFIG_PATH
        (project_root/config/config.yaml) if no config is supplied.

        Args:
            config: A validated PipelineConfig instance.  If None, the
                    default config file is loaded automatically.
        """
        self.grid_data:          Dict[Tuple[int, int], str] = {}
        self.instructions:       Dict[int, Instruction]     = {}
        self.rows:               int                        = 10
        self.cols:               int                        = 10
        self.config:             Optional[PipelineConfig]   = None
        self.bypass_types:       List[BypassType]           = []
        self.rule_checker:       RuleChecker                = RuleChecker()
        self.bypass_annotations: List[BypassAnnotation]     = []

        resolved_config = config or PipelineConfig.load_from_file()
        self.load_config(resolved_config)

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    def load_config(self, config: PipelineConfig) -> None:
        """
        Apply a new PipelineConfig to the scheduler.

        Updates the instruction format registry, bypass types, and rule
        definitions.  Does not clear existing grid_data, instructions, or
        bypass_annotations so that a mid-session config reload preserves
        the user's work.

        If the config specifies a start_state JSON file under
        defaults.start_state, that file is loaded as the initial grid
        state.  Otherwise the default row/col dimensions from the config
        are applied.

        Args:
            config: A validated PipelineConfig instance.
        """
        self.config = config

        # Populate the instruction format registry used by Instruction.__init__
        set_instruction_formats(config.to_instruction_formats())

        # Replace bypass annotation types with those from the new config
        self.bypass_types = config.to_bypass_types()

        # Rebuild the rule set from the new config definition
        self.rule_checker.load_rules_from_config(config)

        # Apply default grid state or dimensions
        default_state = config.get_default_state()
        if default_state:
            self.load_state(default_state)
        else:
            self.rows = config.default_rows
            self.cols = config.default_cols

    # ------------------------------------------------------------------
    # Grid management
    # ------------------------------------------------------------------

    def set_block(self, row: int, col: int, block_type: Optional[str]) -> None:
        """
        Place or clear a block at a grid cell.

        Args:
            row:        Row index (0-based)
            col:        Column index (0-based)
            block_type: Block type string (e.g. 'F', 'D') or None to clear
        """
        if block_type:
            self.grid_data[(row, col)] = block_type
        elif (row, col) in self.grid_data:
            del self.grid_data[(row, col)]

    def resize_grid(self, rows: int, cols: int) -> None:
        """
        Change the visible grid dimensions.

        Blocks outside the new bounds are preserved in grid_data and
        reappear if the grid is later enlarged again.

        Args:
            rows: New number of rows (0–100)
            cols: New number of columns (0–100)
        """
        self.rows = rows
        self.cols = cols

    # ------------------------------------------------------------------
    # Instruction management
    # ------------------------------------------------------------------

    def set_instruction(self, row: int, instruction: Instruction) -> None:
        """
        Assign an instruction to a specific row.

        Args:
            row:         Row index
            instruction: Instruction instance to assign
        """
        self.instructions[row] = instruction

    def get_instruction(self, row: int) -> Optional[Instruction]:
        """
        Retrieve the instruction assigned to a row.

        Args:
            row: Row index

        Returns:
            Optional[Instruction]: The instruction if one is assigned,
                                   None otherwise.
        """
        return self.instructions.get(row)

    # ------------------------------------------------------------------
    # Rule checking
    # ------------------------------------------------------------------

    def check_rules(self) -> List[Dict]:
        """
        Run all enabled rules against the current grid and return violations.

        Returns:
            List[Dict]: Each dict contains:
                rule_name (str):          Name of the violated rule
                cells (list):             [{"row": int, "col": int}, ...]
                rows  (list):             Affected row numbers
                message (str):            Human-readable description
            Empty list if no violations are detected.
        """
        violations = self.rule_checker.check_all(
            self.grid_data,
            self.instructions,
            self.rows,
            self.cols,
        )
        return [
            {
                'rule_name': v.rule_name,
                'cells':     [{'row': c[0], 'col': c[1]} for c in v.cells],
                'rows':      v.rows,
                'message':   v.message,
            }
            for v in violations
        ]

    def get_rules_info(self) -> List[Dict]:
        """
        Return metadata for all registered rules.

        Returns:
            List[Dict]: Each dict contains name, description, and enabled.
        """
        return self.rule_checker.get_rules_info()

    # ------------------------------------------------------------------
    # Bypass types
    # ------------------------------------------------------------------

    def get_bypass_types(self) -> List[Dict]:
        """
        Return all current bypass annotation types as JSON-serializable dicts.

        Returns:
            List[Dict]: Each dict contains name, color, and description.
        """
        return [t.to_dict() for t in self.bypass_types]

    # ------------------------------------------------------------------
    # Bypass annotations
    # ------------------------------------------------------------------

    def add_bypass_annotation(self,
                               annotation_type: str,
                               source:          Tuple[int, int],
                               target:          Tuple[int, int]) -> BypassAnnotation:
        """
        Create, store, and return a new bypass annotation.

        Looks up the colour from self.bypass_types; falls back to blue
        (#0066cc) if annotation_type does not match any registered type.

        Args:
            annotation_type: Name matching a BypassType (e.g. 'RAW')
            source:          (row, col) of the origin cell
            target:          (row, col) of the destination cell

        Returns:
            BypassAnnotation: The newly created annotation.
        """
        matched = next(
            (t for t in self.bypass_types if t.name == annotation_type), None
        )
        color      = matched.color if matched else '#0066cc'
        annotation = BypassAnnotation(annotation_type, color, source, target)
        self.bypass_annotations.append(annotation)
        return annotation

    def remove_bypass_annotations_at(self, row: int, col: int) -> int:
        """
        Remove every annotation whose source or target is the given cell.

        Used by the right-click handler on grid cells.

        Args:
            row: Row index of the cell
            col: Column index of the cell

        Returns:
            int: Number of annotations removed.
        """
        before = len(self.bypass_annotations)
        self.bypass_annotations = [
            a for a in self.bypass_annotations
            if a.source != (row, col) and a.target != (row, col)
        ]
        return before - len(self.bypass_annotations)

    def remove_specific_bypass_annotation(self,
                                           source:          Tuple[int, int],
                                           target:          Tuple[int, int],
                                           annotation_type: str = None) -> int:
        """
        Remove a single annotation identified by source, target, and type.

        More precise than remove_bypass_annotations_at; used by the delete
        button in the annotation list panel where each entry is individually
        identified.

        Args:
            source:          (row, col) of the origin cell
            target:          (row, col) of the destination cell
            annotation_type: If provided, must also match, allowing two
                             annotations with the same endpoints but different
                             types to coexist without conflicting.

        Returns:
            int: Number of annotations removed (0 or 1 under normal usage).
        """
        before = len(self.bypass_annotations)
        self.bypass_annotations = [
            a for a in self.bypass_annotations
            if not (
                a.source == source
                and a.target == target
                and (annotation_type is None or a.annotation_type == annotation_type)
            )
        ]
        return before - len(self.bypass_annotations)

    def get_bypass_annotations(self) -> List[Dict]:
        """
        Return all stored bypass annotations as JSON-serializable dicts.

        Returns:
            List[Dict]: Each dict contains annotation_type, color,
                        source {row, col}, and target {row, col}.
        """
        return [a.to_dict() for a in self.bypass_annotations]

    # ------------------------------------------------------------------
    # State serialization
    # ------------------------------------------------------------------

    def get_state(self) -> Dict:
        """
        Return the complete scheduler state as a JSON-serializable dict.

        Used by GET /api/state and by get_state() route to persist and
        restore the full session.

        Returns:
            dict: Contains grid_data, instructions, rows, cols,
                  config_name, rules, and bypass_annotations.
        """
        return {
            'grid_data': {
                f"{k[0]},{k[1]}": v for k, v in self.grid_data.items()
            },
            'instructions': {
                str(k): v.to_dict() for k, v in self.instructions.items()
            },
            'rows':                self.rows,
            'cols':                self.cols,
            'config_name':         self.config.name if self.config else None,
            'rules':               self.get_rules_info(),
            'bypass_annotations':  self.get_bypass_annotations(),
        }

    def load_state(self, state: Dict) -> None:
        """
        Restore scheduler state from a dict.

        Accepts the same structure as get_state() output, typically parsed
        from a saved JSON file or a POST /api/state request body.

        Does not override the loaded config; only grid, instruction, and
        annotation data are restored.

        Args:
            state: Dict containing any subset of the keys produced by
                   get_state().  Missing keys fall back to empty defaults
                   or config-derived defaults where applicable.
        """
        self.grid_data = {
            tuple(map(int, k.split(','))): v
            for k, v in state.get('grid_data', {}).items()
        }
        self.instructions = {
            int(k): Instruction.from_dict(v)
            for k, v in state.get('instructions', {}).items()
        }
        default_rows = self.config.default_rows if self.config else 10
        default_cols = self.config.default_cols if self.config else 10
        self.rows = state.get('rows', default_rows)
        self.cols = state.get('cols', default_cols)
        self.bypass_annotations = [
            BypassAnnotation.from_dict(a)
            for a in state.get('bypass_annotations', [])
        ]