# pipeline/scheduler.py
from typing import Optional, List, Dict, Tuple
from .instructions import Instruction
from .rules import RuleChecker
from dataclasses import dataclass

@dataclass
class PipelineType:
    """
    Defines a named category of pipeline annotation with a display color.

    Attributes:
        name (str): Short identifier shown in the UI type list (e.g., 'RAW')
        color (str): Hex color string used for drawing circles and arrows
        description (str): Human-readable explanation shown alongside the name
    """
    name: str
    color: str
    description: str

    def to_dict(self) -> Dict:
        return {'name': self.name, 'color': self.color, 'description': self.description}


@dataclass
class PipelineAnnotation:
    """
    Represents a single drawn pipeline annotation connecting two grid cells.

    Attributes:
        annotation_type (str): Matches a PipelineType name
        color (str): Hex color inherited from the PipelineType at creation time
        source (Tuple[int, int]): (row, col) of the circled origin cell
        target (Tuple[int, int]): (row, col) of the arrowhead destination cell
    """
    annotation_type: str
    color: str
    source: Tuple[int, int]
    target: Tuple[int, int]

    def to_dict(self) -> Dict:
        return {
            'annotation_type': self.annotation_type,
            'color': self.color,
            'source': {'row': self.source[0], 'col': self.source[1]},
            'target': {'row': self.target[0], 'col': self.target[1]}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'PipelineAnnotation':
        return cls(
            annotation_type=data['annotation_type'],
            color=data['color'],
            source=(int(data['source']['row']), int(data['source']['col'])),
            target=(int(data['target']['row']), int(data['target']['col']))
        )
class PipelineScheduler:
    """
    Main controller for the CPU instruction pipeline scheduler.

    Manages the grid state, instructions, pipeline configuration,
    and coordinates rule checking. Provides the interface between
    the web frontend and the scheduling logic.
    """
    PIPELINE_TYPES: List[PipelineType] = [
        PipelineType('RAW',        '#cc0000', 'Read After Write'),
        PipelineType('WAR',        '#cc6600', 'Write After Read'),
        PipelineType('WAW',        '#9900cc', 'Write After Write'),
        PipelineType('Control',    '#006600', 'Control Dependency'),
        PipelineType('Structural', '#0066cc', 'Structural Hazard'),
    ]
    def __init__(self):
        self.grid_data = {}
        self.instructions = {}
        self.rows = 10
        self.cols = 10
        self.pipeline_count = 1  # default: 1 pipeline
        self.rule_checker = RuleChecker()
        self.pipeline_annotations: List[PipelineAnnotation] = []

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
    def set_bypass(self, row, col):
        """
        Set a bypass circle on top of the grid
        Args:
            row: Row index
            col: Column index
        """

        self.grid_data[(row, col)] = block_type

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
            'rules': self.get_rules_info(),
            'pipeline_annotations': self.get_pipeline_annotations()
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
        self.pipeline_annotations = [
            PipelineAnnotation.from_dict(a)
            for a in state.get('pipeline_annotations', [])
        ]

    def add_pipeline_annotation(self, annotation_type: str,
                                source: Tuple[int, int],
                                target: Tuple[int, int]) -> PipelineAnnotation:
        """
        Create and store a new pipeline annotation.

        Looks up the color from PIPELINE_TYPES; falls back to blue if
        annotation_type is unrecognised.

        Args:
            annotation_type: Name matching a PipelineType (e.g., 'RAW')
            source: (row, col) of the origin cell
            target: (row, col) of the destination cell

        Returns:
            PipelineAnnotation: The newly created annotation
        """
        matched = next((t for t in self.PIPELINE_TYPES if t.name == annotation_type), None)
        color = matched.color if matched else '#0066cc'
        annotation = PipelineAnnotation(annotation_type, color, source, target)
        self.pipeline_annotations.append(annotation)
        return annotation
    def remove_pipeline_annotations_at(self, row: int, col: int) -> int:
        """
        Remove all annotations whose source or target is the given cell.

        Args:
            row: Row index of the cell
            col: Column index of the cell

        Returns:
            int: Number of annotations removed
        """
        before = len(self.pipeline_annotations)
        self.pipeline_annotations = [
            a for a in self.pipeline_annotations
            if a.source != (row, col) and a.target != (row, col)
        ]
        return before - len(self.pipeline_annotations)

    def get_pipeline_annotations(self) -> List[Dict]:
        """Return all annotations as a list of JSON-serialisable dicts."""
        return [a.to_dict() for a in self.pipeline_annotations]

    @classmethod
    def get_pipeline_types(cls) -> List[Dict]:
        """Return all registered pipeline types as JSON-serialisable dicts."""
        return [t.to_dict() for t in cls.PIPELINE_TYPES]
