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
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Module-level instruction format registry
#
# Stores InstructionFormat objects keyed by instruction mnemonic.
# Empty on module load; populated by set_instruction_formats() when a
# PipelineConfig is loaded by PipelineScheduler.load_config().
# ---------------------------------------------------------------------------

_instruction_formats: Dict[str, 'InstructionFormat'] = {}


def set_instruction_formats(formats: Dict[str, Dict]) -> None:
    """
    Populate the instruction format registry from a config dict.

    Clears any previously registered formats and rebuilds the registry
    from the supplied dict.  Must be called before any Instruction
    instances are created so that their format lookups succeed.

    Called by PipelineScheduler.load_config() each time a new config
    file is applied.

    Args:
        formats: Dict returned by PipelineConfig.to_instruction_formats().
                 Each key is an instruction mnemonic (str); each value is
                 a dict with the keys:
                     'type'     (str):       Instruction type string,
                                             e.g. 'R_TYPE'
                     'operands' (List[str]): Ordered operand names,
                                             e.g. ['rd', 'rs1', 'rs2']
                     'syntax'   (str):       Display format template,
                                             e.g. 'add rd, rs1, rs2'
    """
    global _instruction_formats
    _instruction_formats.clear()
    for name, fmt in formats.items():
        _instruction_formats[name] = InstructionFormat(
            name=name,
            instruction_type=fmt.get('type', ''),
            operands=list(fmt.get('operands', [])),
            syntax=fmt.get('syntax', name),
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
# InstructionFormat
# ---------------------------------------------------------------------------

@dataclass
class InstructionFormat:
    """
    Specification for one instruction type, built from the YAML config.

    Attributes:
        name             (str):       Instruction mnemonic, e.g. 'add'
        instruction_type (str):       One of the strings in
                                      VALID_INSTRUCTION_TYPES, e.g. 'R_TYPE'.
                                      Plain string rather than an enum
                                      (Option A of the refactoring plan).
        operands         (List[str]): Ordered operand names,
                                      e.g. ['rd', 'rs1', 'rs2']
        syntax           (str):       Assembly template string used by
                                      Instruction.to_assembly().
                                      Operand names are substituted with
                                      their concrete values at render time,
                                      e.g. 'lw rd, imm(rs1)'
    """
    name:             str
    instruction_type: str
    operands:         List[str]
    syntax:           str


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