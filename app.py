"""
app.py

Flask application entry point and all route definitions for the CPU
Instruction Pipeline Scheduler.

A single PipelineScheduler instance is created at startup using the config
loaded from project_root/config/config.yaml.  All route handlers delegate
to scheduler methods and contain no scheduling or rule-checking logic.

pipeline_count and its related routes (/api/pipeline-count) have been
removed.  Stage capacities are now defined per-stage in the YAML config.
"""

from flask import Flask, render_template, jsonify, request

from pipeline            import Instruction, PipelineScheduler
from pipeline.config     import PipelineConfig

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Application startup — load config and create scheduler
# ---------------------------------------------------------------------------

_default_config = PipelineConfig.load_from_file()
scheduler       = PipelineScheduler(config=_default_config)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    """
    Render the main application interface.

    Injects instruction format definitions, bypass annotation types, and
    ordered block types into the template so they are available to the
    frontend via window.FLASK_DATA without requiring a separate API call.

    Template variables:
        instruction_formats (dict): Maps instruction mnemonics to their
                                    format spec (type, operands, syntax)
        bypass_types (list):        Bypass annotation type dicts
                                    (name, color, description)
        block_types  (list):        Ordered stage name strings for the
                                    block palette, e.g. ['F','D','I',...]

    Returns:
        str: Rendered HTML template (templates/index.html)
    """
    return render_template(
        'index.html',
        instruction_formats = scheduler.config.to_instruction_formats(),
        bypass_types        = scheduler.get_bypass_types(),
        block_types         = scheduler.config.to_block_types(),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@app.route('/api/state', methods=['GET'])
def get_state():
    """
    Retrieve the complete scheduler state.

    Returns:
        JSON response containing:
            grid_data           (dict):  Maps "row,col" keys to block types
            instructions        (dict):  Maps row numbers to instruction dicts
            rows                (int):   Current number of rows
            cols                (int):   Current number of columns
            config_name         (str):   Name field from the loaded config
            rules               (list):  Rule info dicts
            bypass_annotations  (list):  Annotation dicts
    """
    return jsonify(scheduler.get_state())


@app.route('/api/state', methods=['POST'])
def update_state():
    """
    Overwrite the complete scheduler state.

    Accepts the same structure as GET /api/state.  If a 'rules' key is
    present, the enabled state of each named rule is also updated.

    Request body (JSON): Same structure as GET /api/state response.

    Returns:
        JSON: {"success": true}
    """
    data = request.json
    scheduler.load_state(data)
    if 'rules' in data:
        for r in data['rules']:
            scheduler.rule_checker.set_rule_enabled(
                r.get('name'), r.get('enabled', True)
            )
    return jsonify({'success': True})


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

@app.route('/api/block', methods=['POST'])
def set_block():
    """
    Place or clear a block in a specific grid cell.

    Request body (JSON):
        row        (int):      Row index (0-based)
        col        (int):      Column index (0-based)
        block_type (str|null): Block type string or null to clear the cell

    Returns:
        JSON: {"success": true}
    """
    data       = request.json
    row        = int(data['row'])
    col        = int(data['col'])
    block_type = data.get('block_type')
    scheduler.set_block(row, col, block_type)
    return jsonify({'success': True})


@app.route('/api/resize', methods=['POST'])
def resize():
    """
    Resize the scheduling grid.

    Blocks outside the new bounds are preserved internally and reappear
    if the grid is later enlarged again.

    Request body (JSON):
        rows (int): New number of rows (0–100)
        cols (int): New number of columns (0–100)

    Returns:
        JSON: {"success": true}
    """
    data = request.json
    scheduler.resize_grid(int(data['rows']), int(data['cols']))
    return jsonify({'success': True})


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

@app.route('/api/instruction', methods=['POST'])
def set_instruction():
    """
    Set or update an instruction for a specific row.

    Pass {"name": "", "operands": {}} to clear a row's instruction.

    Request body (JSON):
        row         (int):  Row index (0-based)
        instruction (dict): {"name": str, "operands": {operand: value, ...}}

    Returns:
        JSON: {"success": true}
    """
    data        = request.json
    row         = int(data['row'])
    instruction = Instruction.from_dict(data['instruction'])
    scheduler.set_instruction(row, instruction)
    return jsonify({'success': True})


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

@app.route('/api/check-rules', methods=['GET'])
def check_rules():
    """
    Evaluate all enabled rules against the current grid state.

    Returns:
        JSON: {
            "violations": [
                {
                    "rule_name": str,
                    "cells":     [{"row": int, "col": int}, ...],
                    "rows":      [int, ...],
                    "message":   str
                },
                ...
            ]
        }
        violations is an empty list when the schedule is valid.
    """
    return jsonify({'violations': scheduler.check_rules()})


@app.route('/api/rules', methods=['GET'])
def get_rules():
    """
    List all registered rules and their current enabled state.

    Returns:
        JSON: {
            "rules": [
                {"name": str, "description": str, "enabled": bool},
                ...
            ]
        }
    """
    return jsonify({'rules': scheduler.get_rules_info()})


@app.route('/api/rules', methods=['POST'])
def update_rules():
    """
    Enable or disable pipeline rules.

    Request body — single rule:
        {"name": str, "enabled": bool}

    Request body — all rules at once:
        {"all": bool}

    Returns:
        JSON: {"success": bool}
        HTTP 400 with {"success": false} if the body matches neither format.
    """
    data = request.json
    if 'all' in data:
        scheduler.rule_checker.set_all_enabled(bool(data['all']))
        return jsonify({'success': True})
    if 'name' in data and 'enabled' in data:
        ok = scheduler.rule_checker.set_rule_enabled(
            data['name'], bool(data['enabled'])
        )
        return jsonify({'success': ok})
    return jsonify({'success': False}), 400


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@app.route('/api/config', methods=['GET'])
def get_config():
    """
    Return the currently loaded configuration as JSON.

    Returns:
        JSON: Full config dict from PipelineConfig.to_dict():
              meta, instruction_formats, bypass_types, stage_capacities,
              block_types, rule_definitions, defaults.
    """
    return jsonify(scheduler.config.to_dict())


@app.route('/api/config', methods=['POST'])
def upload_config():
    """
    Accept a .yaml file upload, validate it, and apply it to the scheduler.

    Accepts multipart/form-data with a 'config' file field containing a
    YAML file.  Validation errors are returned as a list in the response
    body rather than raising an exception, so the frontend can display them.

    On success, returns the full config dict (same shape as GET /api/config)
    plus a 'success' key so the frontend can update all data structures
    (block types, instruction formats, bypass types) in a single round trip.

    Returns:
        JSON: config dict + {"success": true}                  HTTP 200
        JSON: {"success": false, "errors": [str, ...]}         HTTP 400
    """
    if 'config' not in request.files:
        return jsonify({'success': False, 'errors': ['No file provided']}), 400

    try:
        config = PipelineConfig.load_from_stream(request.files['config'])
    except Exception as exc:
        return jsonify(
            {'success': False, 'errors': [f'YAML parse error: {exc}']}
        ), 400

    errors = config.validate()
    if errors:
        return jsonify({'success': False, 'errors': errors}), 400

    scheduler.load_config(config)

    response_data            = config.to_dict()
    response_data['success'] = True
    return jsonify(response_data)


# ---------------------------------------------------------------------------
# Bypass types
# ---------------------------------------------------------------------------

@app.route('/api/bypass-types', methods=['GET'])
def get_bypass_types():
    """
    Retrieve all available bypass annotation types from the loaded config.

    Returns:
        JSON: {
            "bypass_types": [
                {"name": str, "color": str, "description": str},
                ...
            ]
        }
    """
    return jsonify({'bypass_types': scheduler.get_bypass_types()})


# ---------------------------------------------------------------------------
# Bypass annotations
# ---------------------------------------------------------------------------

@app.route('/api/bypass-annotations', methods=['GET'])
def get_bypass_annotations():
    """
    Retrieve all currently stored bypass annotations.

    Returns:
        JSON: {
            "annotations": [
                {
                    "annotation_type": str,
                    "color":           str,
                    "source":          {"row": int, "col": int},
                    "target":          {"row": int, "col": int}
                },
                ...
            ]
        }
    """
    return jsonify({'annotations': scheduler.get_bypass_annotations()})


@app.route('/api/bypass-annotations', methods=['POST'])
def add_bypass_annotation():
    """
    Add a new bypass annotation connecting two grid cells.

    annotation_type must match a name from GET /api/bypass-types.

    Request body (JSON):
        annotation_type (str):  Name of the bypass type (e.g. 'RAW')
        source          (dict): {"row": int, "col": int}
        target          (dict): {"row": int, "col": int}

    Returns:
        JSON: {"success": true, "annotation": {annotation_type, color,
               source, target}}
    """
    data   = request.json
    source = (int(data['source']['row']), int(data['source']['col']))
    target = (int(data['target']['row']), int(data['target']['col']))
    annotation = scheduler.add_bypass_annotation(
        data['annotation_type'], source, target
    )
    return jsonify({'success': True, 'annotation': annotation.to_dict()})


@app.route('/api/bypass-annotations', methods=['DELETE'])
def remove_bypass_annotation():
    """
    Remove bypass annotation(s).

    Mode 1 — specific annotation (used by the annotation list delete button):
        Request body: {
            "source":          {"row": int, "col": int},
            "target":          {"row": int, "col": int},
            "annotation_type": str   (optional; narrows to one type)
        }

    Mode 2 — all annotations touching a cell (used by right-click on grid):
        Request body: {"row": int, "col": int}

    Returns:
        JSON: {"success": true, "removed": int}
              removed is the count of annotations deleted.
    """
    data = request.json
    if 'source' in data and 'target' in data:
        source = (int(data['source']['row']), int(data['source']['col']))
        target = (int(data['target']['row']), int(data['target']['col']))
        removed = scheduler.remove_specific_bypass_annotation(
            source, target, data.get('annotation_type')
        )
    else:
        removed = scheduler.remove_bypass_annotations_at(
            int(data['row']), int(data['col'])
        )
    return jsonify({'success': True, 'removed': removed})


if __name__ == '__main__':
    app.run(debug=True, port=5000)