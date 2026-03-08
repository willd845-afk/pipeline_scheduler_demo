# pipeline/scheduler.py
from typing import Optional, List, Dict, Tuple
from .instructions import Instruction
from .rules import RuleChecker

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