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
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import yaml

VALID_INSTRUCTION_TYPES: frozenset = frozenset({
    'R_TYPE', 'I_TYPE', 'I_STORE', 'B_TYPE', 'J_TYPE', 'JR_TYPE',
})

# ---------------------------------------------------------------------------
# Stage string pattern
#
# Matches 'F(1)', 'i*(0)', 'Y0(1)', 'r*(1)', etc.
# Group 1: base name (letters + digits)
# Group 2: optional '*' suffix  — retained as part of the stage name
# Group 3: capacity integer
# ---------------------------------------------------------------------------
_STAGE_STR_RE = re.compile(r'^([A-Za-z][A-Za-z0-9]*)(\*)?\((\d+)\)$')


def _parse_stage_str(s: str) -> Tuple[str, int]:
    """
    Parse a stage entry string of the form 'Name(capacity)' or 'Name*(capacity)'.

    The optional '*' suffix is incorporated into the returned name:
        'F(1)'   → ('F',  1)
        'i*(0)'  → ('i*', 0)   # capacity 0 = unbounded
        'Y0(1)'  → ('Y0', 1)
        'r*(1)'  → ('r*', 1)

    Args:
        s: Raw stage string from the YAML config.

    Returns:
        Tuple[str, int]: (stage_name, capacity)

    Raises:
        ValueError: If the string does not match the expected pattern.
    """
    s = s.strip()
    m = _STAGE_STR_RE.match(s)
    if m:
        name     = m.group(1) + (m.group(2) or '')
        capacity = int(m.group(3))
        return name, capacity
    raise ValueError(
        f"Cannot parse stage string '{s}': expected 'Name(capacity)'"
        f" or 'Name*(capacity)', e.g. 'F(1)' or 'i*(0)'"
    )


# ---------------------------------------------------------------------------
# StageSequenceGraph
# ---------------------------------------------------------------------------

class StageSequenceGraph:
    """
    Directed acyclic graph of all valid pipeline stage sequences.

    Built from the pipeline.stages section by
    PipelineConfig._build_stage_sequence_graph().  The graph models
    sequential ordering within each stage group and parallel-path
    semantics between sub-groups of the same group number.

    Example for the reference config:
        Group 1:  F → D → i* → I     (sequential)
        Group 2:  I → {Y0→Y1→Y2→Y3, X, L0→L1, S}  (parallel paths)
        Group 3:  {Y3,X,L1,S} → W → r* → C  (converging then sequential)

    Attributes:
        nodes        (List[str]):                All stage names, in order of
                                                  first appearance.
        edges        (Dict[str, List[str]]):      stage → [valid next stages]
        predecessors (Dict[str, List[str]]):      stage → [preceding stages]
        stage_depth  (Dict[str, int]):            stage → minimum BFS depth
                                                  from root stages
        group_of     (Dict[str, int]):            stage → YAML group number
        subgroup_of  (Dict[str, Optional[str]]): stage → subgroup key or None
        valid_paths  (List[List[str]]):           All root-to-leaf paths
    """

    def __init__(
            self,
            nodes:        List[str],
            edges:        Dict[str, List[str]],
            predecessors: Dict[str, List[str]],
            stage_depth:  Dict[str, int],
            group_of:     Dict[str, int],
            subgroup_of:  Dict[str, Optional[str]],
            valid_paths:  List[List[str]],
    ) -> None:
        self.nodes        = nodes
        self.edges        = edges
        self.predecessors = predecessors
        self.stage_depth  = stage_depth
        self.group_of     = group_of
        self.subgroup_of  = subgroup_of
        self.valid_paths  = valid_paths

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Return a JSON-serialisable summary of the graph."""
        return {
            'nodes':       self.nodes,
            'edges':       self.edges,
            'stage_depth': self.stage_depth,
            'valid_paths': self.valid_paths,
        }

    # ── Query helpers ───────────────────────────────────────────────────────

    def get_stage_depth(self, stage: str) -> Optional[int]:
        """Return the minimum pipeline depth of *stage*, or None if unknown."""
        return self.stage_depth.get(stage)

    def is_valid_transition(self, from_stage: str, to_stage: str) -> bool:
        """Return True if *to_stage* is an immediate successor of *from_stage*."""
        return to_stage in self.edges.get(from_stage, [])

    def is_reachable(self, from_stage: str, to_stage: str) -> bool:
        """
        Return True if *to_stage* is reachable from *from_stage*.

        Uses BFS over self.edges.  A stage is not considered reachable
        from itself (returns False when from_stage == to_stage).
        """
        if from_stage == to_stage:
            return False
        visited = set()
        queue   = deque([from_stage])
        while queue:
            current = queue.popleft()
            if current == to_stage:
                return True
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self.edges.get(current, []))
        return False

# ---------------------------------------------------------------------------
# Config file path
# Resolved relative to the project root (two directories above this file).
# Expected on disk at:  <project_root>/config/config.yaml
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH   = os.path.join(_PROJECT_ROOT, 'config', 'config.yaml')

# Directory containing config.yaml and any state JSON files
_CONFIG_DIR = os.path.join(_PROJECT_ROOT, 'config')

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
        """
        self._data: Dict = data

        self.meta:           Dict[str, str] = data.get('meta', {})
        self._instructions:  Dict           = data.get('instructions', {})
        self._bypass_types:  List           = data.get('bypass_types', [])  # old format
        self._bypass:        Dict           = data.get('bypass', {})        # new format
        self._pipeline:      Dict           = data.get('pipeline', {})
        self._defaults:      Dict           = data.get('defaults', {})
        self._rules:         List           = data.get('rules', [])

        # Lazily populated by to_stage_sequence_graph()
        self._stage_graph_cache: Optional[StageSequenceGraph] = None
        self._stage_graph_built: bool                         = False

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
        """
        Validate the instructions block.

        New format requires only 'operands'.  'type' and 'format'/'syntax'
        are optional — 'type' is validated only when present, so both old
        and new instruction formats are accepted.
        """
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

            if 'operands' not in instr:
                errors.append(
                    f"Instruction '{name}': missing required field 'operands'"
                )
            elif not isinstance(instr['operands'], list) or not instr['operands']:
                errors.append(
                    f"Instruction '{name}': 'operands' must be a non-empty list"
                )

            # Validate 'type' only when the field is explicitly present (old format)
            if 'type' in instr and instr['type'] not in VALID_INSTRUCTION_TYPES:
                errors.append(
                    f"Instruction '{name}': unrecognised type '{instr['type']}'; "
                    f"valid values are {sorted(VALID_INSTRUCTION_TYPES)}"
                )

        return errors

    def _validate_bypass_types(self) -> List[str]:
        """
        Validate bypass type definitions.

        Accepts both formats:
        - Old: top-level bypass_types list with {name, color, description} entries
        - New: bypass.colors dict mapping name → hex-color string

        Neither section being present is non-blocking (only a warning), since
        configs that use only ordering/capacity rules need no bypass types.
        """
        errors: List[str] = []

        # Old format
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

        # New format
        colors = self._bypass.get('colors', {})
        if isinstance(colors, dict):
            for bt_name, bt_color in colors.items():
                if not isinstance(bt_color, str):
                    errors.append(
                        f"bypass.colors['{bt_name}']: color must be a hex string, "
                        f"got {type(bt_color).__name__}"
                    )

        return errors

    def _validate_pipeline_stages(self) -> List[str]:
        """
        Validate the pipeline.stages block for either format.

        Old format: stage_name → integer capacity
        New format: group_number → list of 'Name(capacity)' strings
                    or          → dict of subgroup_name → list of strings
        """
        errors: List[str] = []
        stages = self._pipeline.get('stages', {})

        if not stages:
            errors.append("'pipeline.stages' block is empty or missing")
            return errors

        is_new_format = any(
            isinstance(k, int) or (isinstance(k, str) and str(k).isdigit())
            for k in stages.keys()
        )

        if is_new_format:
            for raw_key, group_val in stages.items():
                gname = f"pipeline.stages[{raw_key}]"
                if isinstance(group_val, list):
                    for entry in group_val:
                        try:
                            _parse_stage_str(str(entry))
                        except ValueError as exc:
                            errors.append(f"{gname}: {exc}")
                elif isinstance(group_val, dict):
                    for sg_name, sg_list in group_val.items():
                        if not isinstance(sg_list, list):
                            errors.append(
                                f"{gname}.{sg_name}: subgroup value must be"
                                f" a list, got {type(sg_list).__name__}"
                            )
                            continue
                        for entry in sg_list:
                            try:
                                _parse_stage_str(str(entry))
                            except ValueError as exc:
                                errors.append(f"{gname}.{sg_name}: {exc}")
                else:
                    errors.append(
                        f"{gname}: value must be a list or a mapping, "
                        f"got {type(group_val).__name__}"
                    )
        else:
            # Old format
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
        """
        Validate the rules block.

        Only 'name' and 'description' are universally required.
        'enabled' defaults to True when absent (new format omits it).
        'type' is optional — rules may use when/then blocks instead.
        """
        errors: List[str] = []

        for i, rule in enumerate(self._rules):
            if not isinstance(rule, dict):
                errors.append(
                    f"rules[{i}]: must be a mapping, "
                    f"got {type(rule).__name__}"
                )
                continue
            for required in ('name', 'description'):
                if required not in rule:
                    errors.append(
                        f"rules[{i}]: missing required field '{required}'"
                    )

        return errors

    def _validate_start_state(self) -> List[str]:
        """
        Validate the start_state field type only.

        File existence and JSON validity are non-blocking and are reported
        as warnings via get_warnings() rather than blocking errors.
        The only blocking error here is if start_state is set to a non-string
        value (e.g. a dict or list), which indicates a malformed config.

        Returns:
            List[str]: One error string if start_state is the wrong type;
                       empty list otherwise.
        """
        errors: List[str] = []
        start_state = self._defaults.get('start_state')

        # Absent or explicitly null — perfectly valid
        if start_state is None or start_state is False:
            return errors

        if not isinstance(start_state, str):
            errors.append(
                f"defaults.start_state must be a string path, "
                f"got {type(start_state).__name__}"
            )

        return errors

    def get_warnings(self) -> List[str]:
        """
        Return non-blocking warnings about the config state.

        Unlike validate(), warnings do not prevent the config from loading.
        They are surfaced to the user via the terminal at startup and via
        the API response on Import Config so the user can investigate without
        the application refusing to start.

        Current checks:
            - start_state defined but the file is missing or unreadable
            - start_state defined but the file contains invalid JSON
            - JSON files found in the config directory that are not
              referenced by start_state (potential forgotten state files)

        Returns:
            List[str]: Warning strings in the order they were found.
                       Empty list if no issues are detected.
        """
        found: List[str] = []
        found.extend(self._warn_start_state())
        found.extend(self._warn_unreferenced_json())
        return found

    def _warn_start_state(self) -> List[str]:
        """
        Return warnings related to the start_state path.

        Produces a warning when start_state is defined but:
        - the referenced file does not exist on disk, or
        - the referenced file exists but contains invalid JSON.

        In both cases the scheduler will fall back to an empty grid at
        the default dimensions.  No warning is produced when start_state
        is absent or null.

        Returns:
            List[str]: Zero, one, or two warning strings.
        """
        warnings_out: List[str] = []
        start_state = self._defaults.get('start_state')

        if not start_state or not isinstance(start_state, str):
            return warnings_out

        resolved = self._resolve_state_path(start_state)

        if not os.path.exists(resolved):
            warnings_out.append(
                f"start_state '{start_state}' is defined in defaults but "
                f"the file does not exist at '{resolved}'. "
                f"The grid will start empty."
            )
            return warnings_out

        try:
            with open(resolved, 'r', encoding='utf-8') as fh:
                json.load(fh)
        except json.JSONDecodeError as exc:
            warnings_out.append(
                f"start_state '{start_state}' exists but contains invalid "
                f"JSON ({exc}). The grid will start empty."
            )

        return warnings_out

    def _warn_unreferenced_json(self) -> List[str]:
        """
        Return warnings for JSON files in the config directory that are
        not referenced by defaults.start_state.

        Scans _CONFIG_DIR (project_root/config/) for .json files and
        compares each one against the resolved start_state path.  Any
        .json file that exists in the directory but is not referenced
        produces a warning so the user can decide whether to wire it up
        as a start_state or delete it.

        Silently skips the scan if the config directory does not exist or
        cannot be read.

        Returns:
            List[str]: One warning string per unreferenced .json file,
                       sorted alphabetically.
        """
        warnings_out: List[str] = []

        # Resolve the referenced start_state path to an absolute path
        start_state = self._defaults.get('start_state')
        referenced_abs: Optional[str] = None
        if start_state and isinstance(start_state, str):
            referenced_abs = self._resolve_state_path(start_state)

        # Scan the config directory
        try:
            all_files = os.listdir(_CONFIG_DIR)
        except (FileNotFoundError, PermissionError):
            return warnings_out

        json_files = sorted(f for f in all_files if f.lower().endswith('.json'))

        for json_file in json_files:
            full_abs = os.path.abspath(os.path.join(_CONFIG_DIR, json_file))

            # Normalise both sides to avoid false positives from path style
            # differences (e.g. ./config vs config on POSIX)
            if referenced_abs and os.path.normcase(full_abs) == os.path.normcase(referenced_abs):
                continue  # This is the referenced start_state — no warning

            display_path = os.path.join('config', json_file)
            warnings_out.append(
                f"'{display_path}' is a JSON file in the config directory "
                f"but is not referenced by defaults.start_state. "
                f"Add 'start_state: ./{display_path}' to defaults if you "
                f"want it to be loaded automatically."
            )

        return warnings_out

    @staticmethod
    def _resolve_state_path(path: str) -> str:
        """
        Resolve a start_state path to an absolute path.

        Relative paths are resolved relative to the project root
        (_PROJECT_ROOT) so that './config/foo.json' and 'config/foo.json'
        both resolve correctly regardless of the working directory Flask
        is launched from.

        Args:
            path: The raw start_state string from the YAML.

        Returns:
            str: Absolute path string.
        """
        if os.path.isabs(path):
            return path
        return os.path.abspath(os.path.join(_PROJECT_ROOT, path))

    # ------------------------------------------------------------------
    # Pipeline structure parsing
    # ------------------------------------------------------------------

    def _parse_pipeline_structure(
            self,
    ) -> Tuple[List[Tuple[int, Optional[str], str, int]], bool]:
        """
        Parse pipeline.stages in either the old or the new grouped format.

        Old format:  { stage_name: capacity_int, … }
        New format:  { group_num: [Name(cap), …] | {subgroup: [Name(cap), …]} }

        Returns:
            Tuple of:
              - List of (group_num, subgroup_or_None, stage_name, capacity)
                in pipeline traversal order.
              - bool: True when the new grouped/annotated format was detected.
        """
        stages = self._pipeline.get('stages', {})
        all_stages: List[Tuple[int, Optional[str], str, int]] = []

        if not stages:
            return all_stages, False

        is_new_format = any(
            isinstance(k, int) or (isinstance(k, str) and str(k).isdigit())
            for k in stages.keys()
        )

        if not is_new_format:
            # Old format: flat {name: capacity}
            for stage_name, stage_val in stages.items():
                capacity = self._extract_capacity(stage_val)
                all_stages.append((1, None, str(stage_name), capacity or 1))
            return all_stages, False

        # New format: {group_num: list | subgroup_dict}
        for raw_key, group_val in stages.items():
            group_num = int(raw_key)

            if isinstance(group_val, list):
                for stage_str in group_val:
                    try:
                        name, cap = _parse_stage_str(str(stage_str))
                        all_stages.append((group_num, None, name, cap))
                    except ValueError:
                        pass  # Already flagged by _validate_pipeline_stages

            elif isinstance(group_val, dict):
                for subgroup_name, subgroup_list in group_val.items():
                    if isinstance(subgroup_list, list):
                        for stage_str in subgroup_list:
                            try:
                                name, cap = _parse_stage_str(str(stage_str))
                                all_stages.append(
                                    (group_num, str(subgroup_name), name, cap)
                                )
                            except ValueError:
                                pass

        return all_stages, True

    @staticmethod
    def _build_stage_sequence_graph(
            all_stages: List[Tuple[int, Optional[str], str, int]],
    ) -> 'StageSequenceGraph':
        """
        Build a StageSequenceGraph from the parsed stage tuple list.

        Algorithm:
          1. Group tuples by (group_num, subgroup).
          2. Add sequential directed edges within each subgroup (A→B→C…).
          3. Add inter-group edges from the last stage of every subgroup in
             group N to the first stage of every subgroup in group N+1,
             producing parallel-path semantics for multi-subgroup groups.
          4. BFS from root nodes (no predecessors) to compute minimum-depth
             stage indices used by the ordering rule.
          5. DFS from roots to enumerate all complete root-to-leaf paths.

        Args:
            all_stages: List of (group_num, subgroup_or_None, stage_name,
                        capacity) in pipeline traversal order.

        Returns:
            StageSequenceGraph
        """
        # ── 1. Collect stages by group / subgroup ─────────────────────
        groups: Dict[int, Dict[Optional[str], List[str]]] = {}
        for group_num, subgroup, name, _cap in all_stages:
            groups.setdefault(group_num, {}).setdefault(subgroup, []).append(name)

        sorted_group_nums: List[int] = sorted(groups.keys())

        edges: Dict[str, List[str]] = {}
        predecessors: Dict[str, List[str]] = {}
        group_of: Dict[str, int] = {}
        subgroup_of: Dict[str, Optional[str]] = {}

        # ── 2. Intra-subgroup sequential edges ────────────────────────
        for group_num, subgroup_dict in groups.items():
            for sg_name, stage_list in subgroup_dict.items():
                for stage in stage_list:
                    group_of[stage] = group_num
                    subgroup_of[stage] = sg_name
                for i in range(len(stage_list) - 1):
                    s1, s2 = stage_list[i], stage_list[i + 1]
                    edges.setdefault(s1, []).append(s2)
                    predecessors.setdefault(s2, []).append(s1)

        # ── 3. Inter-group edges (last of group N → first of group N+1) ─
        for idx in range(len(sorted_group_nums) - 1):
            curr_gnum = sorted_group_nums[idx]
            next_gnum = sorted_group_nums[idx + 1]

            curr_lasts = [sl[-1] for sl in groups[curr_gnum].values() if sl]
            next_firsts = [sl[0] for sl in groups[next_gnum].values() if sl]

            for last_s in curr_lasts:
                for first_s in next_firsts:
                    edges.setdefault(last_s, []).append(first_s)
                    predecessors.setdefault(first_s, []).append(last_s)

        # ── 4. BFS depths (minimum distance from pipeline entry) ──────
        nodes = list(group_of.keys())
        root_nodes = [n for n in nodes if not predecessors.get(n)]

        stage_depth: Dict[str, int] = {}
        bfs_q: deque = deque((n, 0) for n in root_nodes)
        while bfs_q:
            node, depth = bfs_q.popleft()
            if node not in stage_depth:
                stage_depth[node] = depth
                for succ in edges.get(node, []):
                    bfs_q.append((succ, depth + 1))

        # ── 5. All complete paths (DFS root → leaf) ───────────────────
        valid_paths: List[List[str]] = []

        def _dfs(node: str, path: List[str]) -> None:
            path.append(node)
            successors = edges.get(node, [])
            if not successors:
                valid_paths.append(list(path))
            else:
                for s in successors:
                    _dfs(s, path)
            path.pop()

        for root in root_nodes:
            _dfs(root, [])

        return StageSequenceGraph(
            nodes=nodes,
            edges=edges,
            predecessors=predecessors,
            stage_depth=stage_depth,
            group_of=group_of,
            subgroup_of=subgroup_of,
            valid_paths=valid_paths,
        )

    # ------------------------------------------------------------------
    # Data accessors — used by other pipeline modules and Flask routes
    # ------------------------------------------------------------------

    def to_instruction_formats(self) -> Dict[str, Dict]:
        """
        Return instruction definitions for instructions.py and the template.

        Extends the old shape with new fields from the declarative config
        ('valid_path', 'when_done', 'read_dependency', 'write_dependency').
        Missing fields are returned as empty strings for forward compatibility.

        Returns:
            dict mapping instruction mnemonic to:
                'type'             (str):       Instruction type or ''
                'operands'         (List[str]): Ordered operand names
                'syntax'           (str):       Assembly template
                'valid_path'       (str):       Pipeline path e.g. 'ALU'
                'when_done'        (str):       Stage where result is ready
                'read_dependency'  (str):       Dependency category
                'write_dependency' (str):       Dependency category
        """
        return {
            name: {
                'type':             instr.get('type', ''),
                'operands':         instr.get('operands', []),
                'syntax':           instr.get('format', instr.get('syntax', name)),
                'valid_path':       instr.get('valid_path', ''),
                'when_start':       instr.get('when_start', ''),
                'when_done':        instr.get('when_done', ''),
                'values_committed': instr.get('values_committed', ''),
                'decoded':          instr.get('decoded', ''),
                'read_dependency':  instr.get('read_dependency', ''),
                'write_dependency': instr.get('write_dependency', ''),
            }
            for name, instr in self._instructions.items()
        }

    def to_bypass_types(self) -> List['BypassType']:
        """
        Build BypassType objects from either the new bypass.colors dict or
        the old bypass_types list.

        New format (bypass.colors): name and color taken from the dict;
        description defaults to the name.
        Old format (bypass_types list): all three fields read directly.

        Returns:
            List[BypassType]: Empty list if neither section is present.
        """
        # ── New format: bypass.colors ─────────────────────────────────
        colors_dict = self._bypass.get('colors', {})
        if colors_dict and isinstance(colors_dict, dict):
            return [
                BypassType(name=str(k), color=str(v), description=str(k))
                for k, v in colors_dict.items()
                if isinstance(v, str)
            ]

        # ── Old format: top-level bypass_types list ───────────────────
        return [
            BypassType(
                name=bt['name'],
                color=bt['color'],
                description=bt.get('description', bt['name']),
            )
            for bt in self._bypass_types
            if isinstance(bt, dict)
               and 'name' in bt
               and 'color' in bt
        ]

    def to_stage_capacities(self) -> Dict[str, int]:
        """
        Return a mapping of stage name → absolute column capacity.

        Handles both the old flat dict format and the new grouped
        Name(capacity) notation.  Capacity 0 means unbounded.

        Returns:
            Dict[str, int]: e.g. {'F': 1, 'D': 1, 'i*': 0, 'I': 1, 'X': 1, …}
        """
        all_stages, is_new = self._parse_pipeline_structure()
        if is_new:
            return {name: cap for _g, _sg, name, cap in all_stages}

        # Old format fallback
        stages = self._pipeline.get('stages', {})
        result: Dict[str, int] = {}
        for stage_name, stage_val in stages.items():
            capacity = self._extract_capacity(stage_val)
            result[stage_name] = capacity if capacity is not None else 1
        return result

    def to_block_types(self) -> List[str]:
        """
        Return an ordered list of stage name strings for the block palette.

        New-format order: group 1 stages, then all group 2 sub-groups in
        declaration order, then group 3 stages.

        Returns:
            List[str]: e.g. ['F','D','i*','I','Y0','Y1','Y2','Y3',
                              'X','L0','L1','S','W','r*','C']
        """
        all_stages, is_new = self._parse_pipeline_structure()
        if is_new:
            return [name for _g, _sg, name, _cap in all_stages]
        return list(self._pipeline.get('stages', {}).keys())

    def to_stage_sequence_graph(self) -> Optional['StageSequenceGraph']:
        """
        Build and return the stage sequence graph (new format only).

        The graph is built lazily on the first call and cached; subsequent
        calls return the cached instance.

        Returns:
            StageSequenceGraph: Fully populated graph, or None if the config
            uses the old flat pipeline stage format, which does not encode
            stage ordering.
        """
        if not self._stage_graph_built:
            all_stages, is_new = self._parse_pipeline_structure()
            if is_new and all_stages:
                self._stage_graph_cache = (
                    PipelineConfig._build_stage_sequence_graph(all_stages)
                )
            else:
                self._stage_graph_cache = None
            self._stage_graph_built = True

        return self._stage_graph_cache

    def to_rule_definitions(self) -> List[Dict]:
        """
        Return the raw rule dicts for the rule interpreter in rules.py.

        Unchanged from the original — rules are returned as-is so
        RuleChecker can inspect the 'type' field and dispatch accordingly.
        """
        return list(self._rules)

    def to_dict(self) -> Dict:
        """
        Return a full JSON-serialisable representation of the config.

        Includes 'stage_graph' (new format only) and synthesises
        'bypass_types' from bypass.colors when the old list is absent.

        Returns:
            dict with keys:
                meta, instruction_formats, bypass_types, stage_capacities,
                block_types, rule_definitions, defaults, stage_graph
        """
        graph = self.to_stage_sequence_graph()

        # Normalise bypass_types for the frontend (both formats → list-of-dicts)
        bt_list = [bt for bt in self._bypass_types if isinstance(bt, dict)]
        if not bt_list:
            colors = self._bypass.get('colors', {})
            bt_list = [
                {'name': k, 'color': v, 'description': k}
                for k, v in colors.items()
                if isinstance(v, str)
            ]

        return {
            'meta': self.meta,
            'instruction_formats': self.to_instruction_formats(),
            'bypass_types': bt_list,
            'stage_capacities': self.to_stage_capacities(),
            'block_types': self.to_block_types(),
            'rule_definitions': self.to_rule_definitions(),
            'defaults': {
                'rows': self.default_rows,
                'cols': self.default_cols,
            },
            'stage_graph': graph.to_dict() if graph else None,
        }

    def get_default_state(self) -> Optional[Dict]:
        """
        Load and return the JSON state file referenced by defaults.start_state.

        Uses _resolve_state_path so that relative paths work correctly
        regardless of the working directory.

        Returns:
            dict:  Parsed state dict if the path is set and readable.
            None:  If start_state is absent, the file does not exist, or
                   the file contains invalid JSON.
        """
        start_state = self._defaults.get('start_state')
        if not start_state or not isinstance(start_state, str):
            return None

        resolved = self._resolve_state_path(start_state)

        try:
            with open(resolved, 'r', encoding='utf-8') as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

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