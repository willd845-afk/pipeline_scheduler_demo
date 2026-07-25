"""
pipeline/instructions.py

Defines the data structures for CPU instructions and a module-level
registry that maps instruction names to their format specifications.

The registry is populated at runtime by calling set_instruction_formats()
with data from a PipelineConfig instance.  This replaces the previous
approach of hard-coding all instruction definitions as a class variable
on Instruction.

InstructionType has been removed as an enum (Option A from the refactoring
plan).  The instruction_type field on InstructionFormat is now a plain
string (e.g. 'R_TYPE', 'I_TYPE').  Valid strings are defined in
pipeline/config.py as VALID_INSTRUCTION_TYPES.  All type comparisons in
to_assembly() use string equality rather than enum membership tests.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Module-level instruction format registry
#
# Stores InstructionFormat objects keyed by instruction mnemonic.
# Empty on module load; populated by set_instruction_formats() when a
# PipelineConfig is loaded by PipelineScheduler.load_config().
# ---------------------------------------------------------------------------

_instruction_formats: Dict[str, 'InstructionFormat'] = {}


@dataclass
class InstructionFormat:
    """
    Specification for one instruction type.

    Attributes:
        name             (str): Instruction mnemonic
        operands         (List[str]): Ordered operand names
        instruction_type (str): Type string e.g. 'R_TYPE', or ''
        syntax           (str): Assembly template; auto-generated if empty
        valid_path       (str): Pipeline sub-path e.g. 'ALU', 'MUL'
        when_start       (str): Stage where execution begins e.g. 'X'
        when_done        (str): Stage where result is available for
                                 forwarding e.g. 'X'
        values_committed (str): Stage where result is committed to the
                                 architectural register file e.g. 'C'.
                                 Used as the producer's hazard boundary
                                 when no bypass is present.
        decoded          (str): Stage where the instruction is decoded and
                                 its operands become addressable e.g. 'D'.
                                 Used as the lower bound for valid bypass
                                 targets — a bypass to a stage before this
                                 point cannot be used because the instruction
                                 is not yet ready to receive forwarded data.
        read_dependency  (str): Dependency category for reads e.g. 'reg_update'
        write_dependency (str): Dependency category for writes e.g. 'reg_update'
    """
    name:             str
    operands:         List[str]
    instruction_type: str = ''
    syntax:           str = ''
    valid_path:       str = ''
    when_start:       str = ''
    when_done:        str = ''
    values_committed: str = ''
    decoded:          str = ''
    read_dependency:  str = ''
    write_dependency: str = ''

    def __post_init__(self):
        if not self.syntax:
            self.syntax = (
                f"{self.name} {', '.join(self.operands)}"
                if self.operands else self.name
            )


# ---------------------------------------------------------------------------
# Register helpers
#
# Placed here (rather than rules.py) so that both rules.py and
# dependency_graph.py can import them without creating a circular dependency.
# ---------------------------------------------------------------------------

def _get_written_reg_values(instruction: Any) -> Set[str]:
    """
    Return the set of concrete register values written by *instruction*.

    Operand names starting with 'rd' are treated as write destinations.

    Args:
        instruction: An Instruction instance, or None / any object without
                     a 'format' attribute.

    Returns:
        Set[str]: e.g. {'r1'}.  Empty set if the instruction is None or
                  has no recognised write operands.
    """
    if not instruction or not getattr(instruction, 'format', None):
        return set()
    result: Set[str] = set()
    for op_name in instruction.format.operands:
        if op_name.startswith('rd'):
            val = instruction.operands.get(op_name, '').strip()
            if val:
                result.add(val)
    return result


def _get_read_reg_values(instruction: Any) -> Set[str]:
    """
    Return the set of concrete register values read by *instruction*.

    Operand names starting with 'rs' or 'rt', or exactly matching 'base'
    or 'addr', are treated as read sources.

    Args:
        instruction: An Instruction instance, or None.

    Returns:
        Set[str]: e.g. {'r2', 'r3'}.  Empty set if None or no read operands.
    """
    if not instruction or not getattr(instruction, 'format', None):
        return set()
    _PREFIXES = ('rs', 'rt')
    _EXACT    = frozenset({'base', 'addr'})
    result: Set[str] = set()
    for op_name in instruction.format.operands:
        if op_name.startswith(_PREFIXES) or op_name in _EXACT:
            val = instruction.operands.get(op_name, '').strip()
            if val:
                result.add(val)
    return result


def set_instruction_formats(formats: Dict[str, Dict]) -> None:
    """
    Populate the instruction format registry from a config dict.

    Args:
        formats: Dict returned by PipelineConfig.to_instruction_formats().
    """
    global _instruction_formats
    _instruction_formats.clear()
    for name, fmt in formats.items():
        operands = list(fmt.get('operands', []))
        _instruction_formats[name] = InstructionFormat(
            name=name,
            operands=operands,
            instruction_type=fmt.get('type', ''),
            syntax=fmt.get('syntax', ''),
            valid_path=fmt.get('valid_path', ''),
            when_start=fmt.get('when_start', ''),
            when_done=fmt.get('when_done', ''),
            values_committed=fmt.get('values_committed', ''),
            decoded=fmt.get('decoded', ''),
            read_dependency=fmt.get('read_dependency', ''),
            write_dependency=fmt.get('write_dependency', ''),
        )


def get_instruction_formats() -> Dict[str, 'InstructionFormat']:
    """
    Return the current instruction format registry.

    Returns:
        Dict[str, InstructionFormat]: Maps mnemonic to format object.
                                      Empty dict if set_instruction_formats()
                                      has not been called yet.
    """
    return _instruction_formats

# ---------------------------------------------------------------------------
# Instruction
# ---------------------------------------------------------------------------

class Instruction:
    """
    Represents one CPU instruction with its concrete operand values.

    The format specification is looked up from the module-level registry
    at construction time, so set_instruction_formats() must be called
    before creating Instruction instances.

    If the mnemonic is not found in the registry (e.g. because the config
    has not been loaded yet or the mnemonic is unknown), self.format is
    set to None and to_assembly() falls back to returning the bare mnemonic.
    """

    def __init__(self, name: str, operands: Dict[str, str] = None):
        """
        Create an instruction instance.

        Args:
            name:     Instruction mnemonic (e.g. 'add', 'lw').
                      Looked up in _instruction_formats at construction time.
            operands: Dict mapping operand name to its concrete value,
                      e.g. {'rd': 'r1', 'rs1': 'r2', 'rs2': 'r3'}.
                      Defaults to an empty dict.
        """
        self.name:     str                        = name
        self.format:   Optional[InstructionFormat] = _instruction_formats.get(name)
        self.operands: Dict[str, str]             = operands or {}

    def to_assembly(self) -> str:
        """
        Render the instruction as an assembly string.

        Uses the syntax template from InstructionFormat, replacing each
        operand placeholder with its concrete value from self.operands.
        Operands are substituted longest-name-first to prevent a shorter
        name that is a prefix of a longer one (e.g. 'r' vs 'rs1') from
        being replaced inside the longer name.

        If no format is registered for this mnemonic, returns the bare
        mnemonic string.

        Returns:
            str: Rendered assembly, e.g. 'add r1, r2, r3' or 'lw r1, 4(r2)'
        """
        if not self.format:
            return self.name

        result = self.format.syntax
        # Sort longest-first to avoid partial substitution of overlapping names
        sorted_operands = sorted(self.format.operands, key=len, reverse=True)
        for operand_name in sorted_operands:
            value  = str(self.operands.get(operand_name, operand_name))
            result = result.replace(operand_name, value)
        return result

    def to_dict(self) -> Dict:
        """
        Serialize to a JSON-compatible dict.

        Returns:
            dict: {'name': str, 'operands': dict}
        """
        return {
            'name':     self.name,
            'operands': self.operands,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Instruction':
        """
        Deserialize from a dict (typically from a saved JSON state file).

        Args:
            data: dict containing:
                  'name'     (str):  Instruction mnemonic
                  'operands' (dict): Operand name → value mapping

        Returns:
            Instruction: New instance with format looked up from the
                         current registry state.
        """
        return cls(
            name=data.get('name', ''),
            operands=data.get('operands', {}),
        )