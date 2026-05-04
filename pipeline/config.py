"""
pipeline/config.py

Loads, validates, and exposes all pipeline configuration from a YAML file.

This module is the single source of truth for all data that was previously
hard-coded across instructions.py, rules.py, and scheduler.py:
  - Instruction names, types, operands, and display formats
  - Bypass / hazard annotation types and colours
  - Pipeline stage names and per-stage column capacities
  - Validation rules and their declarative logic definitions
  - Default grid dimensions and optional starting state

BypassType is defined here (rather than in scheduler.py) so that
scheduler.py can import it from config.py without creating a circular
dependency (config.py imports nothing from the pipeline package).

NOTE: pipeline_count and its associated UI sliders have been removed.
Stage capacity is now defined per stage in the YAML file and is the
absolute column limit for that stage regardless of pipeline count.
"""

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import yaml

# ---------------------------------------------------------------------------
# Config file path
# Resolved relative to the project root (two directories above this file).
# Expected on disk at:  <project_root>/config/config.yaml
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH   = os.path.join(_PROJECT_ROOT, 'config', 'config.yaml')

# ---------------------------------------------------------------------------
# Allowed instruction type strings
# Option A: plain strings rather than an enum so that instruction_type is a
# simple str field on InstructionFormat.  All type comparisons in
# instructions.py use string equality (e.g. if itype == 'I_TYPE': ...).
# ---------------------------------------------------------------------------
VALID_INSTRUCTION_TYPES: frozenset = frozenset({
    'R_TYPE',
    'I_TYPE',
    'I_STORE',
    'B_TYPE',
    'J_TYPE',
    'JR_TYPE',
})


# ---------------------------------------------------------------------------
# BypassType dataclass
# Moved here from scheduler.py so scheduler.py can import it from config.py
# without creating a circular dependency.
# ---------------------------------------------------------------------------

@dataclass
class BypassType:
    """
    Defines one bypass / hazard annotation category.

    Instances are built by PipelineConfig.to_bypass_types() from the
    bypass_types block in the YAML.  Scheduler and frontend both consume
    the same objects so display colour is consistent.

    Attributes:
        name        (str): Short identifier shown in the UI, e.g. 'RAW'
        color       (str): Hex colour string, e.g. '#cc0000'
        description (str): Full label, e.g. 'Read After Write'
    """
    name:        str
    color:       str
    description: str

    def to_dict(self) -> Dict:
        """Serialize to a JSON-compatible dict."""
        return {
            'name':        self.name,
            'color':       self.color,
            'description': self.description,
        }


# ---------------------------------------------------------------------------
# StageConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class StageConfig:
    """
    Represents one pipeline stage entry from the pipeline.stages block.

    Attributes:
        name     (str): Stage identifier displayed on the palette,
                        e.g. 'F', 'D', 'Y0', 'X'
        capacity (int): Maximum number of times this stage may appear in a
                        single column.  0 means unbounded — the stage is
                        exempt from the column-constraint rule entirely.
    """
    name:     str
    capacity: int

    def to_dict(self) -> Dict:
        """Serialize to a JSON-compatible dict."""
        return {'name': self.name, 'capacity': self.capacity}


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class PipelineConfig:
    """
    Loads, validates, and exposes all configuration data from a YAML file.

    Typical usage
    -------------
    On startup (uses the hard-coded CONFIG_PATH):
        config = PipelineConfig.load_from_file()
        errors = config.validate()
        if errors:
            raise ValueError('\\n'.join(errors))
        scheduler.load_config(config)

    On a user-uploaded config (from a Flask route):
        config = PipelineConfig.load_from_stream(request.files['config'])
        errors = config.validate()
        if errors:
            return jsonify({'success': False, 'errors': errors}), 400
        scheduler.load_config(config)

    YAML structure expected
    -----------------------
    meta:
      name: "..."
      description: "..."

    instructions:
      add:
        type: R_TYPE
        operands: [rd, rs1, rs2]
        format: "add rd, rs1, rs2"
      ...

    bypass_types:
      - name: RAW
        color: "#cc0000"
        description: "Read After Write"
      ...

    pipeline:
      stages:
        F: 1          # simple form:   stage_name: capacity
        i: 0          # 0 = unbounded
        Y0: 1

    defaults:
      rows: 10
      cols: 10
      start_state: './config/start_example.json'   # optional

    rules:
      - name: "..."
        description: "..."
        enabled: true
        type: "column_constraint"
        logic: { ... }
    """

    def __init__(self, data: Dict):
        """
        Wrap a pre-parsed YAML dict.

        Prefer the class methods (load_from_file, load_from_stream,
        load_from_dict) over calling this constructor directly.

        Args:
            data: A dict produced by yaml.safe_load() or equivalent.
                  Missing top-level keys default to empty containers so
                  all accessor methods are safe to call on partial configs.
        """
        self._data: Dict = data

        self.meta:           Dict[str, str] = data.get('meta', {})
        self._instructions:  Dict           = data.get('instructions', {})
        self._bypass_types:  List           = data.get('bypass_types', [])
        self._pipeline:      Dict           = data.get('pipeline', {})
        self._defaults:      Dict           = data.get('defaults', {})
        self._rules:         List           = data.get('rules', [])

    # ------------------------------------------------------------------
    # Classmethods — preferred constructors
    # ------------------------------------------------------------------

    @classmethod
    def load_from_file(cls, path: str = None) -> 'PipelineConfig':
        """
        Load config from a YAML file on disk.

        Uses CONFIG_PATH (project_root/config/config.yaml) by default.

        Args:
            path: Absolute or relative path to the YAML file.
                  Pass None to use the hard-coded CONFIG_PATH.

        Returns:
            PipelineConfig: New instance wrapping the parsed data.

        Raises:
            FileNotFoundError: If the file does not exist at the given path.
            yaml.YAMLError:    If the file content is not valid YAML.
        """
        if path is None:
            path = CONFIG_PATH
        with open(path, 'r', encoding='utf-8') as fh:
            data = yaml.safe_load(fh)
        return cls.load_from_dict(data or {})

    @classmethod
    def load_from_stream(cls, stream) -> 'PipelineConfig':
        """
        Load config from a file-like object.

        Intended for Flask file upload handling:
            config = PipelineConfig.load_from_stream(request.files['config'])

        Args:
            stream: Any object accepted by yaml.safe_load() — typically a
                    Werkzeug FileStorage stream or a BytesIO / StringIO object.

        Returns:
            PipelineConfig: New instance wrapping the parsed data.

        Raises:
            yaml.YAMLError: If the stream content is not valid YAML.
        """
        data = yaml.safe_load(stream)
        return cls.load_from_dict(data or {})

    @classmethod
    def load_from_dict(cls, data: Dict) -> 'PipelineConfig':
        """
        Wrap an already-parsed dict as a PipelineConfig.

        Useful for testing or for constructing configs programmatically.

        Args:
            data: A dict whose structure matches the expected YAML layout.

        Returns:
            PipelineConfig: New instance.
        """
        return cls(data)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """
        Check all config sections for required fields and valid values.

        Does not raise an exception; instead returns a list of error strings
        so the caller can decide whether to abort or display warnings.

        Returns:
            List[str]: One string per problem found.
                       An empty list means the config is valid.
        """
        errors: List[str] = []
        errors.extend(self._validate_instructions())
        errors.extend(self._validate_bypass_types())
        errors.extend(self._validate_pipeline_stages())
        errors.extend(self._validate_rules())
        errors.extend(self._validate_start_state())
        return errors

    def _validate_instructions(self) -> List[str]:
        """Validate the instructions block."""
        errors: List[str] = []

        if not self._instructions:
            errors.append("'instructions' block is empty or missing")
            return errors

        for name, instr in self._instructions.items():
            if not isinstance(instr, dict):
                errors.append(
                    f"Instruction '{name}': must be a mapping, "
                    f"got {type(instr).__name__}"
                )
                continue

            # type
            if 'type' not in instr:
                errors.append(
                    f"Instruction '{name}': missing required field 'type'"
                )
            elif instr['type'] not in VALID_INSTRUCTION_TYPES:
                errors.append(
                    f"Instruction '{name}': invalid type '{instr['type']}'; "
                    f"must be one of {sorted(VALID_INSTRUCTION_TYPES)}"
                )

            # operands
            if 'operands' not in instr:
                errors.append(
                    f"Instruction '{name}': missing required field 'operands'"
                )
            elif not isinstance(instr['operands'], list) or not instr['operands']:
                errors.append(
                    f"Instruction '{name}': 'operands' must be a non-empty list"
                )

            # format
            if 'format' not in instr:
                errors.append(
                    f"Instruction '{name}': missing required field 'format'"
                )

        return errors

    def _validate_bypass_types(self) -> List[str]:
        """Validate the bypass_types block. Missing block is allowed (returns empty)."""
        errors: List[str] = []

        for i, bt in enumerate(self._bypass_types):
            if not isinstance(bt, dict):
                errors.append(
                    f"bypass_types[{i}]: must be a mapping, "
                    f"got {type(bt).__name__}"
                )
                continue
            for required in ('name', 'color', 'description'):
                if required not in bt:
                    errors.append(
                        f"bypass_types[{i}]: missing required field '{required}'"
                    )

        return errors

    def _validate_pipeline_stages(self) -> List[str]:
        """Validate the pipeline.stages block."""
        errors: List[str] = []
        stages = self._pipeline.get('stages', {})

        if not stages:
            errors.append("'pipeline.stages' block is empty or missing")
            return errors

        for stage_name, stage_val in stages.items():
            capacity = self._extract_capacity(stage_val)
            if capacity is None:
                errors.append(
                    f"Stage '{stage_name}': capacity must be a non-negative "
                    f"integer, got '{stage_val}'"
                )
            elif capacity < 0:
                errors.append(
                    f"Stage '{stage_name}': capacity must be >= 0, "
                    f"got {capacity}"
                )

        return errors

    def _validate_rules(self) -> List[str]:
        """Validate the rules block."""
        errors: List[str] = []

        for i, rule in enumerate(self._rules):
            if not isinstance(rule, dict):
                errors.append(
                    f"rules[{i}]: must be a mapping, "
                    f"got {type(rule).__name__}"
                )
                continue
            for required in ('name', 'description', 'enabled', 'type'):
                if required not in rule:
                    errors.append(
                        f"rules[{i}]: missing required field '{required}'"
                    )

        return errors

    def _validate_start_state(self) -> List[str]:
        """
        Validate defaults.start_state if present.
        Checks that the file exists and contains valid JSON.
        """
        errors: List[str] = []
        start_state = self._defaults.get('start_state')

        if not start_state:
            return errors

        if not os.path.exists(start_state):
            errors.append(
                f"defaults.start_state: path '{start_state}' does not exist"
            )
            return errors

        try:
            with open(start_state, 'r', encoding='utf-8') as fh:
                json.load(fh)
        except json.JSONDecodeError as exc:
            errors.append(
                f"defaults.start_state: '{start_state}' is not valid "
                f"JSON — {exc}"
            )

        return errors

    # ------------------------------------------------------------------
    # Data accessors — used by other pipeline modules and Flask routes
    # ------------------------------------------------------------------

    def to_instruction_formats(self) -> Dict[str, Dict]:
        """
        Return instruction definitions in the shape expected by instructions.py
        and by the Flask template variable injection.

        Replaces the hard-coded Instruction.FORMATS class variable.

        Returns:
            dict: Maps instruction name (str) to a dict with keys:
                  'type'     (str):       Instruction type string, e.g. 'R_TYPE'
                  'operands' (List[str]): Ordered operand names, e.g. ['rd','rs1','rs2']
                  'syntax'   (str):       Display format string, e.g. 'add rd, rs1, rs2'
        """
        return {
            name: {
                'type':     instr.get('type', ''),
                'operands': instr.get('operands', []),
                'syntax':   instr.get('format', name),
            }
            for name, instr in self._instructions.items()
        }

    def to_bypass_types(self) -> List[BypassType]:
        """
        Build BypassType objects from the bypass_types block.

        Replaces the hard-coded PipelineScheduler.PIPELINE_TYPES class variable.

        Returns:
            List[BypassType]: One entry per valid bypass_type in the YAML.
                                Empty list if the bypass_types block is absent.
        """
        return [
            BypassType(
                name=bt['name'],
                color=bt['color'],
                description=bt['description'],
            )
            for bt in self._bypass_types
            if isinstance(bt, dict)
            and 'name'        in bt
            and 'color'       in bt
            and 'description' in bt
        ]

    def to_stage_capacities(self) -> Dict[str, int]:
        """
        Return a mapping of stage name to absolute column capacity.

        A capacity of 0 means the stage is unbounded and is exempt from
        column-constraint rule checking.  There is no pipeline_count
        multiplier; the value from the YAML is the direct limit.

        Returns:
            Dict[str, int]: e.g. {'F': 1, 'D': 1, 'i': 0, 'I': 1, ...}
        """
        stages = self._pipeline.get('stages', {})
        result: Dict[str, int] = {}
        for stage_name, stage_val in stages.items():
            capacity = self._extract_capacity(stage_val)
            # Fall back to 1 (strict) rather than 0 (unbounded) on parse failure
            result[stage_name] = capacity if capacity is not None else 1
        return result

    def to_block_types(self) -> List[str]:
        """
        Return an ordered list of stage name strings for building the palette.

        Order is preserved from the YAML definition, so the palette reflects
        the intended stage ordering of the config author.

        Returns:
            List[str]: e.g. ['F', 'D', 'i', 'I', 'Y0', 'Y1', 'Y2', 'Y3',
                              'W', 'r', 'C', 'X']
        """
        return list(self._pipeline.get('stages', {}).keys())

    def to_rule_definitions(self) -> List[Dict]:
        """
        Return the raw rule dicts for the rule interpreter in rules.py.

        Each dict is the complete YAML rule entry including the logic block.
        RuleChecker.load_rules_from_config() inspects the 'type' field of
        each dict to decide which Rule subclass to instantiate.

        Returns:
            List[Dict]: One dict per rule defined in the YAML.
        """
        return list(self._rules)

    def get_default_state(self) -> Optional[Dict]:
        """
        Load and return the JSON state file referenced by defaults.start_state.

        The JSON file uses the same format as the Save State / Load State
        feature (grid_data, instructions, rows, cols, pipeline_annotations).

        Returns:
            dict:  Parsed state dict if the path is set and readable.
            None:  If defaults.start_state is absent, the file does not exist,
                   or the file contains invalid JSON.  Callers should treat
                   None as "use empty defaults".
        """
        start_state = self._defaults.get('start_state')
        if not start_state:
            return None

        try:
            with open(start_state, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def to_dict(self) -> Dict:
        """
        Return a full JSON-serializable representation of the config.

        Used by the POST /api/config route to return everything the frontend
        needs in a single response so it can update block types, instruction
        formats, pipeline types, and rule info in one round trip.

        Returns:
            dict with keys:
                meta                (dict):        name and description strings
                instruction_formats (dict):        output of to_instruction_formats()
                bypass_types        (List[dict]):  raw bypass_type entries
                stage_capacities    (dict):        output of to_stage_capacities()
                block_types         (List[str]):   output of to_block_types()
                rule_definitions    (List[dict]):  output of to_rule_definitions()
                defaults            (dict):        rows (int) and cols (int)
        """
        return {
            'meta':                self.meta,
            'instruction_formats': self.to_instruction_formats(),
            'bypass_types': [
                bt for bt in self._bypass_types
                if isinstance(bt, dict)
            ],
            'stage_capacities':  self.to_stage_capacities(),
            'block_types':       self.to_block_types(),
            'rule_definitions':  self.to_rule_definitions(),
            'defaults': {
                'rows': self.default_rows,
                'cols': self.default_cols,
            },
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable config name from the meta block."""
        return self.meta.get('name', 'Unnamed Configuration')

    @property
    def description(self) -> str:
        """Config description from the meta block."""
        return self.meta.get('description', '')

    @property
    def default_rows(self) -> int:
        """Row count to use when no saved state is present. Defaults to 10."""
        return int(self._defaults.get('rows', 10))

    @property
    def default_cols(self) -> int:
        """Column count to use when no saved state is present. Defaults to 10."""
        return int(self._defaults.get('cols', 10))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_capacity(stage_val: Any) -> Optional[int]:
        """
        Parse an integer capacity from a stage definition value.

        The YAML supports two formats for a stage entry:

            Simple (integer directly):
                F: 1

            Extended (mapping with a capacity key):
                F:
                  capacity: 1

        Args:
            stage_val: The raw value associated with a stage name key in
                       the pipeline.stages dict.

        Returns:
            int:  The capacity value if it could be parsed as a non-negative
                  integer.
            None: If the value is neither an int nor a dict containing
                  an integer 'capacity' key.
        """
        if isinstance(stage_val, int):
            return stage_val
        if isinstance(stage_val, dict):
            cap = stage_val.get('capacity')
            if isinstance(cap, int):
                return cap
        return None