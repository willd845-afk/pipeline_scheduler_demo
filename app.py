# app.py
from flask import Flask, render_template, jsonify, request
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Set, Tuple
from abc import ABC, abstractmethod

app = Flask(__name__)

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
              instructions: Dict[int, Instruction],
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
                  instructions: Dict[int, Instruction],
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

class PipelineScheduler:
    """
    Main controller for the CPU instruction pipeline scheduler.

    Manages the grid state, instructions, pipeline configuration,
    and coordinates rule checking. Provides the interface between
    the web frontend and the scheduling logic.
    """
    def __init__(self):
        self.grid_data = {}
        self.instructions = {}
        self.rows = 10
        self.cols = 10
        self.pipeline_count = 1  # default: 1 pipeline
        self.rule_checker = RuleChecker()

    def set_block(self, row, col, block_type):
        """
        Set or clear a block in the grid.

        Args:
            row: Row index
            col: Column index
            block_type: Block type string (e.g., 'F', 'D') or None to clear
        """
        if block_type:
            self.grid_data[(row, col)] = block_type
        elif (row, col) in self.grid_data:
            del self.grid_data[(row, col)]

    def set_instruction(self, row: int, instruction: Instruction):
        """
        Assign an instruction to a specific row.

        Args:
            row: Row index
            instruction: Instruction instance to assign
        """
        self.instructions[row] = instruction

    def get_instruction(self, row: int) -> Optional[Instruction]:
        """
        Retrieve the instruction assigned to a row.

        Args:
            row: Row index

        Returns:
            Optional[Instruction]: The instruction if one exists, None otherwise
        """
        return self.instructions.get(row)

    def resize_grid(self, rows, cols):
        """
        Resize the scheduling grid.

        Args:
            rows: New number of rows
            cols: New number of columns
        """
        self.rows = rows
        self.cols = cols

    def set_pipeline_count(self, count: int):
        """
        Set the number of concurrent pipelines.

        Args:
            count: Number of pipelines (must be 1 or 2)
        """
        if count in (1, 2):
            self.pipeline_count = count

    def check_rules(self) -> List[Dict]:
        """
        Check all rules and return violations in JSON-serializable format.

        Returns:
            List[Dict]: List of violation dictionaries with rule_name, cells, rows, and message
        """
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
        """
        Get information about all registered rules.

        Returns:
            List[Dict]: List of rule information dictionaries
        """
        return self.rule_checker.get_rules_info()

    def get_state(self):
        """
        Get the complete scheduler state in JSON-serializable format.

        Returns:
            dict: State dictionary containing grid_data, instructions, dimensions,
                  pipeline_count, and rules
        """
        return {
            'grid_data': {f"{k[0]},{k[1]}": v for k, v in self.grid_data.items()},
            'instructions': {str(k): v.to_dict() for k, v in self.instructions.items()},
            'rows': self.rows,
            'cols': self.cols,
            'pipeline_count': self.pipeline_count,
            'rules': self.get_rules_info()
        }

    def load_state(self, state):
        """
        Load scheduler state from a dictionary.

        Args:
            state: State dictionary (typically from JSON)
        """
        self.grid_data = {tuple(map(int, k.split(','))): v for k, v in state.get('grid_data', {}).items()}
        self.instructions = {int(k): Instruction.from_dict(v) for k, v in state.get('instructions', {}).items()}
        self.rows = state.get('rows', 10)
        self.cols = state.get('cols', 10)
        self.pipeline_count = state.get('pipeline_count', self.pipeline_count)

scheduler = PipelineScheduler()

@app.route('/')
def index():
    """
    Render the main application interface.

    Serves the primary HTML template with instruction format definitions,
    current rules information, and pipeline count injected for client-side use.

    Template Variables:
        instruction_formats (dict): Maps instruction names to their format specifications
                                   (type, operands, syntax)
        rules_info (list): List of dictionaries containing rule name, description,
                          and enabled status
        pipeline_count (int): Current number of pipelines (1 or 2)

    Returns:
        str: Rendered HTML template
    """
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
    """
    Retrieve the complete scheduler state.

    Returns the current state of the pipeline scheduler including grid data,
    instructions, dimensions, pipeline count, and rules configuration.
    Used for state persistence and UI synchronization.

    Returns:
        JSON response containing:
            grid_data (dict): Maps "row,col" keys to block types
            instructions (dict): Maps row numbers (as strings) to instruction objects
            rows (int): Current number of rows
            cols (int): Current number of columns
            pipeline_count (int): Number of pipelines (1 or 2)
            rules (list): List of rule information dictionaries

    Example Response:
        {
            "grid_data": {"0,0": "F", "0,1": "D"},
            "instructions": {"0": {"name": "add", "operands": {"rd": "r1", "rs1": "r2", "rs2": "r3"}}},
            "rows": 10,
            "cols": 10,
            "pipeline_count": 1,
            "rules": [{"name": "...", "description": "...", "enabled": true}]
        }
    """
    return jsonify(scheduler.get_state())

@app.route('/api/state', methods=['POST'])
def update_state():
    """
    Update the complete scheduler state from a saved configuration.

    Accepts a full state object (typically from a saved JSON file) and
    restores the scheduler to that state. This includes grid data, instructions,
    dimensions, pipeline count, and rule configurations.

    Request Body (JSON):
        grid_data (dict, optional): Grid cell mappings
        instructions (dict, optional): Instruction configurations per row
        rows (int, optional): Number of rows
        cols (int, optional): Number of columns
        pipeline_count (int, optional): Number of pipelines
        rules (list, optional): Rule configurations with enabled states

    Returns:
        JSON response:
            success (bool): True if state was loaded successfully

    Example Request:
        {
            "grid_data": {"0,0": "F"},
            "instructions": {"0": {"name": "add", "operands": {...}}},
            "rows": 10,
            "cols": 10,
            "pipeline_count": 2,
            "rules": [{"name": "...", "enabled": true}]
        }
    """
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
    """
    Set or clear a block in a specific grid cell.

    Places a pipeline stage block (F, D, I, X, Y0-Y3, W, r, C) at the specified
    grid coordinates, or clears the cell if block_type is null/None.

    Request Body (JSON):
        row (int): Row index (0-based)
        col (int): Column index (0-based)
        block_type (str|null): Block type identifier or null to clear

    Returns:
        JSON response:
            success (bool): True if operation completed

    Example Request:
        {
            "row": 0,
            "col": 5,
            "block_type": "F"
        }
    """
    data = request.json
    row = int(data['row'])
    col = int(data['col'])
    block_type = data.get('block_type')
    scheduler.set_block(row, col, block_type)
    return jsonify({'success': True})

@app.route('/api/instruction', methods=['POST'])
def set_instruction():
    """
    Set or update an instruction for a specific row.

    Configures the instruction type and operands for a given row in the
    pipeline schedule. Each row represents one instruction in the schedule.

    Request Body (JSON):
        row (int): Row index (0-based)
        instruction (object): Instruction configuration
            name (str): Instruction mnemonic (e.g., 'add', 'lw', 'bne')
            operands (dict): Maps operand names to values
                            (e.g., {"rd": "r1", "rs1": "r2", "rs2": "r3"})

    Returns:
        JSON response:
            success (bool): True if operation completed

    Example Request:
        {
            "row": 0,
            "instruction": {
                "name": "add",
                "operands": {
                    "rd": "r1",
                    "rs1": "r2",
                    "rs2": "r3"
                }
            }
        }
    """
    data = request.json
    row = int(data['row'])
    instruction = Instruction.from_dict(data['instruction'])
    scheduler.set_instruction(row, instruction)
    return jsonify({'success': True})

@app.route('/api/resize', methods=['POST'])
def resize():
    """
    Resize the scheduling grid dimensions.

    Changes the number of rows and/or columns in the pipeline grid.
    Existing blocks outside the new dimensions are preserved in the
    internal state but won't be visible until the grid is enlarged again.

    Request Body (JSON):
        rows (int): New number of rows (0-100)
        cols (int): New number of columns (0-100)

    Returns:
        JSON response:
            success (bool): True if operation completed

    Example Request:
        {
            "rows": 15,
            "cols": 20
        }
    """
    data = request.json
    rows = int(data['rows'])
    cols = int(data['cols'])
    scheduler.resize_grid(rows, cols)
    return jsonify({'success': True})

@app.route('/api/check-rules', methods=['GET'])
def check_rules():
    """
    Check all enabled rules and return violations.

    Evaluates the current pipeline schedule against all enabled rules
    and returns detailed information about any violations, including
    which cells and rows are affected.

    Returns:
        JSON response containing:
            violations (list): List of violation objects, each containing:
                rule_name (str): Name of the violated rule
                cells (list): List of {"row": int, "col": int} objects
                rows (list): List of affected row numbers
                message (str): Human-readable violation description

    Example Response:
        {
            "violations": [
                {
                    "rule_name": "Unique Execution Blocks Per Column",
                    "cells": [{"row": 0, "col": 5}, {"row": 1, "col": 5}],
                    "rows": [0, 1],
                    "message": "Block 'X' appears 2 times in column 5"
                }
            ]
        }
    """
    violations = scheduler.check_rules()
    return jsonify({'violations': violations})

@app.route('/api/rules', methods=['GET'])
def get_rules():
    """
    Retrieve information about all registered rules.

    Returns metadata for all pipeline scheduling rules, including
    their names, descriptions, and current enabled/disabled status.

    Returns:
        JSON response containing:
            rules (list): List of rule objects, each containing:
                name (str): Rule name
                description (str): Detailed rule description
                enabled (bool): Whether rule is currently active

    Example Response:
        {
            "rules": [
                {
                    "name": "Unique Execution Blocks Per Column",
                    "description": "X, Y0, Y1, Y2, and Y3 blocks cannot occupy...",
                    "enabled": true
                },
                {
                    "name": "Pipeline Stage Count Per Column",
                    "description": "F, D, I, W, and C blocks may appear...",
                    "enabled": true
                }
            ]
        }
    """
    rules_info = scheduler.get_rules_info()
    return jsonify({'rules': rules_info})

@app.route('/api/rules', methods=['POST'])
def update_rules():
    """
    Enable or disable pipeline rules.

    Allows toggling individual rules or all rules at once. When rules are
    disabled, they won't be checked during validation, allowing users to
    temporarily bypass specific constraints.

    Request Body (JSON) - Option 1 (single rule):
        name (str): Name of the rule to modify
        enabled (bool): Whether to enable or disable the rule

    Request Body (JSON) - Option 2 (all rules):
        all (bool): Enable (true) or disable (false) all rules

    Returns:
        JSON response:
            success (bool): True if operation completed successfully
            error (str, optional): Error message if request was invalid

    Example Request 1 (single rule):
        {
            "name": "Unique Execution Blocks Per Column",
            "enabled": false
        }

    Example Request 2 (all rules):
        {
            "all": true
        }
    """
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
    """
    Set the number of concurrent pipelines.

    Configures whether the scheduler uses 1 or 2 pipelines. This affects
    how many times certain pipeline stages (F, D, I, W, C) can appear in
    the same column, as enforced by the PipelineStageCountPerColumnRule.

    Request Body (JSON):
        pipeline_count (int): Number of pipelines (must be 1 or 2)

    Returns:
        JSON response:
            success (bool): True if operation completed
            error (str, optional): Error message if invalid count provided

    Status Codes:
        200: Success
        400: Invalid pipeline count (not 1 or 2)

    Example Request:
        {
            "pipeline_count": 2
        }
    """
    data = request.json
    count = int(data.get('pipeline_count', 1))
    if count not in (1, 2):
        return jsonify({'success': False, 'error': 'pipeline_count must be 1 or 2'}), 400
    scheduler.set_pipeline_count(count)
    return jsonify({'success': True})

@app.route('/api/pipeline-count', methods=['GET'])
def get_pipeline_count():
    """
    Retrieve the current number of pipelines.

    Returns the currently configured pipeline count (1 or 2).

    Returns:
        JSON response:
            pipeline_count (int): Current number of pipelines

    Example Response:
        {
            "pipeline_count": 1
        }
    """
    return jsonify({'pipeline_count': scheduler.pipeline_count})

if __name__ == '__main__':
    app.run(debug=True, port=5000)