# pipeline/instructions.py
from enum import Enum
from dataclasses import dataclass
from typing import List

class InstructionType(Enum):
    """
    Enumeration of CPU instruction types.

    Represents the different categories of RISC-V-style instructions
    used in the pipeline scheduler.
    """

    R_TYPE = "R-type"
    I_TYPE = "I-type"
    I_STORE = "I-store"
    B_TYPE = "B-type"
    J_TYPE = "J-type"
    JR_TYPE = "JR-type"

@dataclass
class InstructionFormat:
    """
    Defines the format specification for a CPU instruction.

    Attributes:
        name (str): The instruction mnemonic (e.g., 'add', 'lw')
        instruction_type (InstructionType): The type category of the instruction
        operands (List[str]): List of operand names (e.g., ['rd', 'rs1', 'rs2'])
        syntax (str): Human-readable syntax example
    """
    name: str
    instruction_type: InstructionType
    operands: List[str]
    syntax: str

class Instruction:
    """
    Represents a CPU instruction with its operands.

    Handles instruction creation, assembly generation, and serialization
    for different instruction formats (R-type, I-type, etc.).
    """
    FORMATS = {
        'add': InstructionFormat('add', InstructionType.R_TYPE, ['rd', 'rs1', 'rs2'], 'add rd, rs1, rs2'),
        'addi': InstructionFormat('addi', InstructionType.I_TYPE, ['rd', 'rs1', 'imm'], 'addi rd, rs1, imm'),
        'mul': InstructionFormat('mul', InstructionType.R_TYPE, ['rd', 'rs1', 'rs2'], 'mul rd, rs1, rs2'),
        'bne': InstructionFormat('bne', InstructionType.B_TYPE, ['rs1', 'rs2', 'imm'], 'bne rs1, rs2, imm'),
        'jr': InstructionFormat('jr', InstructionType.JR_TYPE, ['rs1'], 'jr rs1'),
        'lw': InstructionFormat('lw', InstructionType.I_TYPE, ['rd', 'rs1', 'imm'], 'lw rd, imm(rs1)'),
        'sw': InstructionFormat('sw', InstructionType.I_STORE, ['rs2', 'rs1', 'imm'], 'sw rs2, imm(rs1)'),
        'jal': InstructionFormat('jal', InstructionType.J_TYPE, ['rd', 'imm'], 'jal rd, imm'),
    }

    def __init__(self, name: str, operands: dict = None):
        """
        Initialize an instruction instance.

        Args:
            name: The instruction mnemonic
            operands: Dictionary mapping operand names to their values
        """
        self.name = name
        self.format = self.FORMATS.get(name)
        self.operands = operands or {}

    def to_assembly(self) -> str:
        """
        Generate assembly string representation of the instruction.

        Returns:
            str: Assembly language string (e.g., "lw r1, 4(r2)")
        """
        if not self.format:
            return self.name

        parts = [self.name]

        if self.format.instruction_type in [InstructionType.I_TYPE, InstructionType.I_STORE]:
            if self.name in ['lw', 'sw']:
                if self.name == 'lw':
                    rd = self.operands.get('rd', 'rd')
                    rs1 = self.operands.get('rs1', 'rs1')
                    imm = self.operands.get('imm', '0')
                    return f"{self.name} {rd}, {imm}({rs1})"
                else:
                    rs2 = self.operands.get('rs2', 'rs2')
                    rs1 = self.operands.get('rs1', 'rs1')
                    imm = self.operands.get('imm', '0')
                    return f"{self.name} {rs2}, {imm}({rs1})"

        operand_values = []
        for operand_name in self.format.operands:
            operand_values.append(self.operands.get(operand_name, operand_name))

        if operand_values:
            return f"{self.name} {', '.join(operand_values)}"
        return self.name

    def to_dict(self):
        """
        Serialize instruction to dictionary format.

        Returns:
            dict: Dictionary containing name and operands
        """
        return {
            'name': self.name,
            'operands': self.operands
        }

    @classmethod
    def from_dict(cls, data):
        """
        Deserialize instruction from dictionary format.

        Args:
            data: Dictionary containing instruction data

        Returns:
            Instruction: New instruction instance
        """
        return cls(data.get('name', ''), data.get('operands', {}))