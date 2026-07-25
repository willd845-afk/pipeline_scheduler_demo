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
from .dependency_graph import DependencyGraph
import warnings as _warnings

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
        self.dependency_graph: DependencyGraph = DependencyGraph()

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

        If start_state is defined but cannot be loaded (file missing or
        invalid JSON), a UserWarning is issued and the scheduler falls back
        to the default dimensions from the config.

        Args:
            config: A validated PipelineConfig instance.
        """
        self.config = config

        set_instruction_formats(config.to_instruction_formats())
        self.bypass_types = config.to_bypass_types()
        self.rule_checker.load_rules_from_config(config)

        default_state = config.get_default_state()
        if default_state:
            self.load_state(default_state)
        else:
            # Warn if start_state was defined but failed to load
            start_state_path = config._defaults.get('start_state')
            if start_state_path and isinstance(start_state_path, str):
                _warnings.warn(
                    f"PipelineScheduler: start_state '{start_state_path}' is "
                    f"defined but could not be loaded. "
                    f"Starting with an empty grid at default dimensions.",
                    UserWarning,
                    stacklevel=2,
                )
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
        Rebuild the dependency graph then run all enabled rules.

        The dependency graph is always rebuilt here (rather than lazily)
        so it reflects the latest operand values typed by the user.
        """
        self.build_dependency_graph()
        violations = self.rule_checker.check_all(
            self.grid_data,
            self.instructions,
            self.rows,
            self.cols,
            self.bypass_annotations,
            dep_graph=self.dependency_graph,
        )
        return [
            {
                'rule_name': v.rule_name,
                'cells': [{'row': c[0], 'col': c[1]} for c in v.cells],
                'rows': v.rows,
                'message': v.message,
                'annotation_indices': v.annotation_indices,
                'highlight_color': v.highlight_color,
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
    # Graph data
    # ------------------------------------------------------------------

    def get_stage_sequence_graph(self) -> Optional[Dict]:
        """
        Return the stage sequence graph as a JSON-serialisable dict.

        Delegates to PipelineConfig.to_stage_sequence_graph(), which
        builds the graph lazily on first call and caches it.

        Returns:
            dict with keys nodes, edges, stage_depth, valid_paths
            when the config uses the new grouped pipeline.stages format.
            None for old-format configs whose flat stage definition does
            not encode ordering information.
        """
        if self.config is None:
            return None
        graph = self.config.to_stage_sequence_graph()
        return graph.to_dict() if graph else None

    def build_dependency_graph(self) -> Dict:
        """
        Rebuild and return the register dependency graph as a
        JSON-serializable dict.

        Always rebuilds from the current instruction set before returning
        so the caller receives fresh data reflecting the latest operand
        values typed by the user.

        Returns:
            dict with keys:
                nodes (dict): Maps row index string to DepNode data
                              including written_regs, read_regs,
                              when_start, when_done, values_committed,
                              decoded, parents, children.
                edges (list): All RAW / WAW / WAR edges with shared_regs.
        """
        self.dependency_graph.build(self.instructions)
        return self.dependency_graph.to_dict()

    # ── add after build_dependency_graph() ────────────────────────────────────

    def check_bypass_bounds(self, producer_row: int, consumer_row: int) -> Dict:
        """
        Validate every bypass annotation connecting producer_row → consumer_row
        against the three column bounds derived from InstructionFormat attributes.

        Mirrors the (corrected) logic in TemporalBypassExistsCondition so the
        /api/debug/bypass-validation endpoint reports exactly which check fails.

        Bounds for the default RAW config:
            source.col >= col(when_done  of producer)
            target.col >= col(decoded    of consumer)
            target.col <  col(when_start of consumer)
        """
        prod_instr = self.instructions.get(producer_row)
        cons_instr = self.instructions.get(consumer_row)

        prod_fmt = getattr(prod_instr, 'format', None) if prod_instr else None
        cons_fmt = getattr(cons_instr, 'format', None) if cons_instr else None

        source_stage = getattr(prod_fmt, 'when_done', '') if prod_fmt else ''
        target_min_stage = getattr(cons_fmt, 'decoded', '') if cons_fmt else ''
        target_max_stage = getattr(cons_fmt, 'when_start', '') if cons_fmt else ''

        def _find_col(row: int, stage: str) -> Optional[int]:
            if not stage:
                return None
            return next(
                (c for c in range(self.cols)
                 if self.grid_data.get((row, c)) == stage),
                None,
            )

        source_min_col = _find_col(producer_row, source_stage)
        target_min_col = _find_col(consumer_row, target_min_stage)
        target_max_col = _find_col(consumer_row, target_max_stage)

        candidates = []
        for ann in self.bypass_annotations:
            if ann.source[0] != producer_row or ann.target[0] != consumer_row:
                continue

            src_col = ann.source[1]
            tgt_col = ann.target[1]

            check_src = source_min_col is None or src_col >= source_min_col
            check_tgt_min = target_min_col is None or tgt_col >= target_min_col
            check_tgt_max = target_max_col is None or tgt_col < target_max_col

            if not check_src:
                reason = (f"source col {src_col} < {source_min_col} "
                          f"(need >= col of {source_stage})")
            elif not check_tgt_min:
                reason = (f"target col {tgt_col} < {target_min_col} "
                          f"(need >= col of {target_min_stage})")
            elif not check_tgt_max:
                reason = (f"target col {tgt_col} >= {target_max_col} "
                          f"(need < col of {target_max_stage})")
            else:
                reason = None

            candidates.append({
                'source': {'row': ann.source[0], 'col': ann.source[1]},
                'target': {'row': ann.target[0], 'col': ann.target[1]},
                'check_source_gte_min': check_src,
                'check_target_gte_min': check_tgt_min,
                'check_target_lt_max': check_tgt_max,
                'valid': reason is None,
                'failure_reason': reason,
            })

        return {
            'producer_row': producer_row,
            'consumer_row': consumer_row,
            'bounds': {
                'source_stage': source_stage,
                'source_min_col': source_min_col,
                'target_min_stage': target_min_stage,
                'target_min_col': target_min_col,
                'target_max_stage': target_max_stage,
                'target_max_col': target_max_col,
            },
            'candidates': candidates,
            'any_valid': any(c['valid'] for c in candidates),
        }

    def evaluate_rules_trace(self, rule_name: str) -> Optional[Dict]:
        """
        Trace a named dependency_rule binding-by-binding for the debug blueprint.
        Dynamically extracts condition properties from YAML rules to support RAW,
        WAW, and WAR trace validation.
        """
        from .rules import QuantifiedYAMLRule

        target_rule = next(
            (r for r in self.rule_checker.rules if r.name == rule_name),
            None,
        )
        if target_rule is None:
            return None

        if not isinstance(target_rule, QuantifiedYAMLRule):
            return {
                'rule_name': rule_name,
                'bindings_evaluated': 0,
                'violations_emitted': 0,
                'trace': [],
                'note': 'Only dependency_rule types support tracing.',
            }

        self.dependency_graph.build(self.instructions)

        # ── 1. Locate original YAML rule dictionary to extract attributes dynamically ──
        rule_defs = self.config.to_rule_definitions()
        raw_def = next((rd for rd in rule_defs if rd.get('name') == rule_name), {})

        # Recursive search to locate 'temporal_hazard' sub-block inside nested 'all'/'any'/'not' YAML trees
        def _find_temporal_hazard_def(block: Any) -> Optional[Dict]:
            if not isinstance(block, dict):
                return None
            if 'temporal_hazard' in block:
                return block['temporal_hazard']
            for key in ['all', 'any']:
                if key in block and isinstance(block[key], list):
                    for sub in block[key]:
                        found = _find_temporal_hazard_def(sub)
                        if found:
                            return found
            if 'not' in block:
                return _find_temporal_hazard_def(block['not'])
            return None

        hazard_def = _find_temporal_hazard_def(raw_def.get('when')) or {}

        # Extract dynamic properties with fallbacks
        dep_class = target_rule._dep_class or 'RAW'
        prod_stage_attr = hazard_def.get('producer_stage_attr', 'values_committed')
        cons_stage_attr = hazard_def.get('consumer_stage_attr', 'when_start')
        operator_str = hazard_def.get('operator', 'gte')

        _OPS = {
            'gt': lambda a, b: a > b,
            'gte': lambda a, b: a >= b,
            'lt': lambda a, b: a < b,
            'lte': lambda a, b: a <= b,
        }
        op_fn = _OPS.get(operator_str, _OPS['gte'])

        trace = []
        violations_emitted = 0

        for edge in self.dependency_graph.get_edges_by_class(dep_class):
            prod_row = edge.producer_row
            cons_row = edge.consumer_row
            prod_instr = self.instructions.get(prod_row)
            cons_instr = self.instructions.get(cons_row)
            if not prod_instr or not cons_instr:
                continue

            prod_fmt = getattr(prod_instr, 'format', None)
            cons_fmt = getattr(cons_instr, 'format', None)

            prod_stage_name = getattr(prod_fmt, prod_stage_attr, '') if prod_fmt else ''
            cons_stage_name = getattr(cons_fmt, cons_stage_attr, '') if cons_fmt else ''

            def _col(row: int, stage: str) -> Optional[int]:
                if not stage:
                    return None
                return next(
                    (c for c in range(self.cols)
                     if self.grid_data.get((row, c)) == stage),
                    None,
                )

            prod_col = _col(prod_row, prod_stage_name)
            cons_col = _col(cons_row, cons_stage_name)

            if prod_col is None or cons_col is None:
                temporal_fires = False
            else:
                temporal_fires = op_fn(prod_col, cons_col)

            # Re-use our trace validation check
            bypass_info = self.check_bypass_bounds(prod_row, cons_row)
            any_valid = bypass_info['any_valid']
            violation = temporal_fires and not any_valid

            if violation:
                violations_emitted += 1

            trace.append({
                'producer_row': prod_row,
                'consumer_row': cons_row,
                'shared_regs': sorted(edge.shared_regs),
                'temporal_hazard': {
                    'producer_stage': prod_stage_name,
                    'producer_col': prod_col,
                    'consumer_stage': cons_stage_name,
                    'consumer_col': cons_col,
                    'result': temporal_fires,
                },
                'bypass_exists': {
                    'any_valid': any_valid,
                    'annotations_on_pair': [
                        {
                            'source': c['source'],
                            'target': c['target'],
                            'passes_source_bound': c['check_source_gte_min'],
                            'passes_target_min': c['check_target_gte_min'],
                            'passes_target_max': c['check_target_lt_max'],
                        }
                        for c in bypass_info['candidates']
                    ],
                },
                'violation_emitted': violation,
            })

        return {
            'rule_name': rule_name,
            'bindings_evaluated': len(trace),
            'violations_emitted': violations_emitted,
            'trace': trace,
        }

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