# app.py
from flask import Flask, render_template, jsonify, request
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Set, Tuple
from abc import ABC, abstractmethod

app = Flask(__name__)

class InstructionType(Enum):
    R_TYPE = "R-type"
    I_TYPE = "I-type"
    I_STORE = "I-store"
    B_TYPE = "B-type"
    J_TYPE = "J-type"
    JR_TYPE = "JR-type"

@dataclass
class InstructionFormat:
    name: str
    instruction_type: InstructionType
    operands: List[str]
    syntax: str

class Instruction:
    """Represents a CPU instruction with its operands"""

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
        self.name = name
        self.format = self.FORMATS.get(name)
        self.operands = operands or {}

    def to_assembly(self) -> str:
        """Generate assembly string representation"""
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
        return {
            'name': self.name,
            'operands': self.operands
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data.get('name', ''), data.get('operands', {}))

@dataclass
class Violation:
    """Represents a rule violation"""
    rule_name: str
    cells: List[Tuple[int, int]]  # List of (row, col) tuples
    rows: List[int]  # List of row numbers for instruction violations
    message: str

class Rule(ABC):
    """Abstract base class for pipeline rules"""

    def __init__(self, name: str, description: str, enabled: bool = True):
        self.name = name
        self.description = description
        self.enabled = enabled

    @abstractmethod
    def check(self, grid_data: Dict[Tuple[int, int], str],
              instructions: Dict[int, Instruction],
              rows: int, cols: int, pipeline_count: int) -> List[Violation]:
        """Check if the rule is violated. Returns list of violations."""
        pass

class UniqueBlockPerColumnRule(Rule):
    """Rule: Certain blocks (X, Y0, Y1, Y2, Y3) cannot appear more than once in the same column"""

    def __init__(self, enabled: bool = True):
        super().__init__(
            "Unique Execution Blocks Per Column",
            "X, Y0, Y1, Y2, and Y3 blocks cannot occupy the same column more than once",
            enabled=enabled
        )
        self.restricted_blocks = {'X', 'Y0', 'Y1', 'Y2', 'Y3'}

    def check(self, grid_data: Dict[Tuple[int, int], str],
              instructions: Dict[int, Instruction],
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
    """Rule: F, D, I, W, and C steps can only occur as many times in a column as there are pipelines"""

    def __init__(self, enabled: bool = True):
        super().__init__(
            "Pipeline Stage Count Per Column",
            "F, D, I, W, and C blocks may appear in a column at most once per pipeline",
            enabled=enabled
        )
        self.stage_blocks = {'F', 'D', 'I', 'W', 'C'}

    def check(self, grid_data: Dict[Tuple[int, int], str],
              instructions: Dict[int, Instruction],
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
    """Manages and checks all pipeline rules"""

    def __init__(self):
        self.rules: List[Rule] = []
        self._register_default_rules()

    def _register_default_rules(self):
        """Register default rules"""
        self.add_rule(UniqueBlockPerColumnRule(enabled=True))
        self.add_rule(PipelineStageCountPerColumnRule(enabled=True))

    def add_rule(self, rule: Rule):
        """Add a rule to the checker"""
        self.rules.append(rule)

    def check_all(self, grid_data: Dict[Tuple[int, int], str],
                  instructions: Dict[int, Instruction],
                  rows: int, cols: int,
                  pipeline_count: int) -> List[Violation]:
        """Check all enabled rules and return all violations"""
        all_violations = []
        for rule in self.rules:
            if not getattr(rule, 'enabled', True):
                continue
            violations = rule.check(grid_data, instructions, rows, cols, pipeline_count)
            all_violations.extend(violations)
        return all_violations

    def get_rules_info(self) -> List[Dict]:
        """Get information about all registered rules"""
        return [
            {
                'name': rule.name,
                'description': rule.description,
                'enabled': getattr(rule, 'enabled', True)
            }
            for rule in self.rules
        ]

    def set_rule_enabled(self, rule_name: str, enabled: bool) -> bool:
        """Enable or disable a specific rule by name"""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = enabled
                return True
        return False

    def set_all_enabled(self, enabled: bool):
        """Enable or disable all rules"""
        for rule in self.rules:
            rule.enabled = enabled

class PipelineScheduler:
    def __init__(self):
        self.grid_data = {}
        self.instructions = {}
        self.rows = 10
        self.cols = 10
        self.pipeline_count = 1  # default: 1 pipeline
        self.rule_checker = RuleChecker()

    def set_block(self, row, col, block_type):
        if block_type:
            self.grid_data[(row, col)] = block_type
        elif (row, col) in self.grid_data:
            del self.grid_data[(row, col)]

    def set_instruction(self, row: int, instruction: Instruction):
        self.instructions[row] = instruction

    def get_instruction(self, row: int) -> Optional[Instruction]:
        return self.instructions.get(row)

    def resize_grid(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def set_pipeline_count(self, count: int):
        """Set the number of pipelines (1 or 2)"""
        if count in (1, 2):
            self.pipeline_count = count

    def check_rules(self) -> List[Dict]:
        """Check all rules and return violations"""
        violations = self.rule_checker.check_all(
            self.grid_data,
            self.instructions,
            self.rows,
            self.cols,
            self.pipeline_count
        )

        return [
            {
                'rule_name': v.rule_name,
                'cells': [{'row': cell[0], 'col': cell[1]} for cell in v.cells],
                'rows': v.rows,
                'message': v.message
            }
            for v in violations
        ]

    def get_rules_info(self) -> List[Dict]:
        """Get information about all rules"""
        return self.rule_checker.get_rules_info()

    def get_state(self):
        return {
            'grid_data': {f"{k[0]},{k[1]}": v for k, v in self.grid_data.items()},
            'instructions': {str(k): v.to_dict() for k, v in self.instructions.items()},
            'rows': self.rows,
            'cols': self.cols,
            'pipeline_count': self.pipeline_count,
            'rules': self.get_rules_info()
        }

    def load_state(self, state):
        self.grid_data = {tuple(map(int, k.split(','))): v for k, v in state.get('grid_data', {}).items()}
        self.instructions = {int(k): Instruction.from_dict(v) for k, v in state.get('instructions', {}).items()}
        self.rows = state.get('rows', 10)
        self.cols = state.get('cols', 10)
        self.pipeline_count = state.get('pipeline_count', self.pipeline_count)

scheduler = PipelineScheduler()

@app.route('/')
def index():
    instruction_formats = {name: {
        'type': fmt.instruction_type.value,
        'operands': fmt.operands,
        'syntax': fmt.syntax
    } for name, fmt in Instruction.FORMATS.items()}

    rules_info = scheduler.get_rules_info()

    return render_template('index.html',
                           instruction_formats=instruction_formats,
                           rules_info=rules_info,
                           pipeline_count=scheduler.pipeline_count)

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(scheduler.get_state())

@app.route('/api/state', methods=['POST'])
def update_state():
    data = request.json
    scheduler.load_state(data)
    # also update rule enabled states if provided
    if 'rules' in data:
        for r in data['rules']:
            scheduler.rule_checker.set_rule_enabled(r.get('name'), r.get('enabled', True))
    if 'pipeline_count' in data:
        scheduler.set_pipeline_count(int(data['pipeline_count']))
    return jsonify({'success': True})

@app.route('/api/block', methods=['POST'])
def set_block():
    data = request.json
    row = int(data['row'])
    col = int(data['col'])
    block_type = data.get('block_type')
    scheduler.set_block(row, col, block_type)
    return jsonify({'success': True})

@app.route('/api/instruction', methods=['POST'])
def set_instruction():
    data = request.json
    row = int(data['row'])
    instruction = Instruction.from_dict(data['instruction'])
    scheduler.set_instruction(row, instruction)
    return jsonify({'success': True})

@app.route('/api/resize', methods=['POST'])
def resize():
    data = request.json
    rows = int(data['rows'])
    cols = int(data['cols'])
    scheduler.resize_grid(rows, cols)
    return jsonify({'success': True})

@app.route('/api/check-rules', methods=['GET'])
def check_rules():
    violations = scheduler.check_rules()
    return jsonify({'violations': violations})

@app.route('/api/rules', methods=['GET'])
def get_rules():
    rules_info = scheduler.get_rules_info()
    return jsonify({'rules': rules_info})

@app.route('/api/rules', methods=['POST'])
def update_rules():
    data = request.json
    # Expect either {name: ..., enabled: true} or {"all": true/false}
    if 'all' in data:
        enabled = bool(data['all'])
        scheduler.rule_checker.set_all_enabled(enabled)
        return jsonify({'success': True})
    if 'name' in data and 'enabled' in data:
        name = data['name']
        enabled = bool(data['enabled'])
        ok = scheduler.rule_checker.set_rule_enabled(name, enabled)
        return jsonify({'success': ok})
    return jsonify({'success': False}), 400

@app.route('/api/pipeline-count', methods=['POST'])
def set_pipeline_count():
    data = request.json
    count = int(data.get('pipeline_count', 1))
    if count not in (1, 2):
        return jsonify({'success': False, 'error': 'pipeline_count must be 1 or 2'}), 400
    scheduler.set_pipeline_count(count)
    return jsonify({'success': True})

@app.route('/api/pipeline-count', methods=['GET'])
def get_pipeline_count():
    return jsonify({'pipeline_count': scheduler.pipeline_count})

if __name__ == '__main__':
    app.run(debug=True, port=5000)