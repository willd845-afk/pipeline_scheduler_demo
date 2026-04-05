# app.py
from flask import Flask, render_template, jsonify, request
from pipeline import Instruction, PipelineScheduler

app = Flask(__name__)
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
                           pipeline_count=scheduler.pipeline_count,
                           pipeline_types=PipelineScheduler.get_pipeline_types())  # NEW

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

@app.route('/api/pipeline-types', methods=['GET'])
def get_pipeline_types():
    """
    Retrieve all available pipeline annotation types.

    Returns:
        JSON response:
            pipeline_types (list): Each entry has name, color, description
    """
    return jsonify({'pipeline_types': PipelineScheduler.get_pipeline_types()})


@app.route('/api/pipeline-annotations', methods=['GET'])
def get_pipeline_annotations():
    """
    Retrieve all currently stored pipeline annotations.

    Returns:
        JSON response:
            annotations (list): Each entry has annotation_type, color, source, target
    """
    return jsonify({'annotations': scheduler.get_pipeline_annotations()})


@app.route('/api/pipeline-annotations', methods=['POST'])
def add_pipeline_annotation():
    """
    Add a new pipeline annotation between two grid cells.

    Request Body (JSON):
        annotation_type (str): Name of the pipeline type (e.g., 'RAW')
        source (dict): {"row": int, "col": int} – origin cell
        target (dict): {"row": int, "col": int} – destination cell

    Returns:
        JSON response:
            success (bool): True
            annotation (dict): The newly created annotation
    """
    data = request.json
    source = (int(data['source']['row']), int(data['source']['col']))
    target = (int(data['target']['row']), int(data['target']['col']))
    annotation = scheduler.add_pipeline_annotation(data['annotation_type'], source, target)
    return jsonify({'success': True, 'annotation': annotation.to_dict()})


@app.route('/api/pipeline-annotations', methods=['DELETE'])
def remove_pipeline_annotation():
    """
    Remove pipeline annotation(s) from the schedule.

    Supports two removal modes depending on the request body:

    Mode 1 – Specific annotation (used by the annotation list delete button):
        Provide source, target, and annotation_type to remove exactly one entry.

    Mode 2 – Cell-based removal (used by right-click on a grid cell):
        Provide row and col to remove all annotations involving that cell.

    Request Body (JSON) – Mode 1:
        source (dict):          {"row": int, "col": int}
        target (dict):          {"row": int, "col": int}
        annotation_type (str):  Optional; narrows match to one type

    Request Body (JSON) – Mode 2:
        row (int): Row index of the cell
        col (int): Column index of the cell

    Returns:
        JSON response:
            success (bool): True
            removed (int):  Number of annotations deleted
    """
    data = request.json

    if 'source' in data and 'target' in data:
        source = (int(data['source']['row']), int(data['source']['col']))
        target = (int(data['target']['row']), int(data['target']['col']))
        annotation_type = data.get('annotation_type')
        removed = scheduler.remove_specific_pipeline_annotation(source, target, annotation_type)
    else:
        removed = scheduler.remove_pipeline_annotations_at(
            int(data['row']), int(data['col'])
        )

    return jsonify({'success': True, 'removed': removed})

def remove_specific_pipeline_annotation(
        self,
        source: Tuple[int, int],
        target: Tuple[int, int],
        annotation_type: str = None) -> int:
    """
    Remove a single annotation matching source, target, and optionally type.

    More precise than remove_pipeline_annotations_at; used when deleting
    from the annotation list panel where each item is individually identified.

    Args:
        source:          (row, col) of the origin cell
        target:          (row, col) of the destination cell
        annotation_type: If provided, must also match; allows two annotations
                         with the same endpoints but different types to coexist

    Returns:
        int: Number of annotations removed (0 or 1 under normal usage)
    """
    before = len(self.pipeline_annotations)
    self.pipeline_annotations = [
        a for a in self.pipeline_annotations
        if not (
            a.source == source and
            a.target == target and
            (annotation_type is None or a.annotation_type == annotation_type)
        )
    ]
    return before - len(self.pipeline_annotations)

if __name__ == '__main__':
    app.run(debug=True, port=5000)