"""
pipeline/debug_routes.py

Debug Blueprint for the Pipeline Scheduler.

Registered unconditionally in app.py:
    from pipeline.debug_routes import debug_bp, init_debug_blueprint
    init_debug_blueprint(scheduler)
    app.register_blueprint(debug_bp)

All routes are under /api/debug/*.
"""

from flask import Blueprint, jsonify, request

debug_bp = Blueprint('debug', __name__, url_prefix='/api/debug')

_scheduler_instance = None


def init_debug_blueprint(scheduler_instance) -> None:
    """Bind the blueprint to the live scheduler.  Call once from app.py."""
    global _scheduler_instance
    _scheduler_instance = scheduler_instance


def _scheduler():
    if _scheduler_instance is None:
        raise RuntimeError(
            "Debug blueprint accessed before init_debug_blueprint() was called."
        )
    return _scheduler_instance


# ── 1. Instruction format fields ──────────────────────────────────────────────

@debug_bp.route('/instruction-formats')
def instruction_formats():
    """
    Verify YAML fields reached Python: config → to_instruction_formats()
    → set_instruction_formats() → InstructionFormat attributes.
    Any empty string indicates a gap in the pipeline.
    """
    return jsonify(_scheduler().config.to_instruction_formats())


# ── 2. Stage column positions ─────────────────────────────────────────────────

@debug_bp.route('/stage-columns')
def stage_columns():
    """
    For each active instruction row show which column each InstructionFormat
    stage attribute occupies.  'not_placed' means the stage is declared but
    the cell has not been dragged into the grid yet; temporal conditions will
    return matched=False (safe fallback, no false violations) until it is placed.
    """
    sched = _scheduler()
    STAGE_ATTRS = ['when_start', 'when_done', 'values_committed', 'decoded']
    result = {}

    for row, instr in sched.instructions.items():
        if not instr or not getattr(instr, 'name', ''):
            continue
        fmt = getattr(instr, 'format', None)
        if not fmt:
            continue

        row_stages = {}
        for attr in STAGE_ATTRS:
            stage_name = getattr(fmt, attr, '')
            if not stage_name:
                row_stages[attr] = {
                    'stage_name': '',
                    'column':     None,
                    'status':     'not_declared',
                }
                continue

            col = next(
                (c for c in range(sched.cols)
                 if sched.grid_data.get((row, c)) == stage_name),
                None,
            )
            row_stages[attr] = {
                'stage_name': stage_name,
                'column':     col,
                'status':     'ok' if col is not None else 'not_placed',
            }

        result[str(row)] = {'instruction': instr.name, 'stages': row_stages}

    return jsonify(result)


# ── 3. Dependency graph detail ────────────────────────────────────────────────

@debug_bp.route('/dep-graph-detail')
def dep_graph_detail():
    """
    Rebuild the dependency graph and return it with a 'type_match' diagnostic
    on every edge.  type_match=False means a register overlap was detected but
    the dep_type strings did not match, silently suppressing the edge.
    summary.type_mismatches is the count of such suppressed potential edges.
    """
    sched = _scheduler()
    sched.dependency_graph.build(sched.instructions)

    augmented_edges = []
    for edge in sched.dependency_graph.edges:
        edge_dict = edge.to_dict()
        p = sched.dependency_graph.nodes.get(edge.producer_row)
        c = sched.dependency_graph.nodes.get(edge.consumer_row)

        if p and c:
            if edge.dep_class == 'RAW':
                type_match = p.write_dep_type == c.read_dep_type
            elif edge.dep_class == 'WAW':
                type_match = p.write_dep_type == c.write_dep_type
            elif edge.dep_class == 'WAR':
                type_match = p.read_dep_type == c.write_dep_type
            else:
                type_match = None
        else:
            type_match = None

        edge_dict['type_match'] = type_match
        augmented_edges.append(edge_dict)

    # Count pairs where register overlap exists but dep_type mismatch blocked edge creation
    nodes = sched.dependency_graph.nodes
    sorted_rows = sorted(nodes.keys())
    type_mismatches = 0
    for i, prod_row in enumerate(sorted_rows):
        for cons_row in sorted_rows[i + 1:]:
            p = nodes[prod_row]
            c = nodes[cons_row]
            if p.written_regs & c.read_regs and p.write_dep_type != c.read_dep_type:
                type_mismatches += 1

    summary = {
        'total_edges': len(augmented_edges),
        'by_class': {
            'RAW': sum(1 for e in augmented_edges if e['dep_class'] == 'RAW'),
            'WAW': sum(1 for e in augmented_edges if e['dep_class'] == 'WAW'),
            'WAR': sum(1 for e in augmented_edges if e['dep_class'] == 'WAR'),
        },
        'type_mismatches': type_mismatches,
    }

    graph_dict = sched.dependency_graph.to_dict()
    return jsonify({'nodes': graph_dict['nodes'],
                    'edges': augmented_edges,
                    'summary': summary})


# ── 4. Bypass column bound validation ─────────────────────────────────────────

@debug_bp.route('/bypass-validation')
def bypass_validation():
    """
    For every bypass annotation connecting producer_row → consumer_row,
    report pass/fail for each of the three column bound checks.
    failure_reason names the exact check that rejected the annotation.

    Query params: producer_row (int), consumer_row (int)
    """
    try:
        producer_row = int(request.args.get('producer_row', 0))
        consumer_row = int(request.args.get('consumer_row', 1))
    except (TypeError, ValueError):
        return jsonify(
            {'error': 'producer_row and consumer_row must be integers'}
        ), 400

    return jsonify(_scheduler().check_bypass_bounds(producer_row, consumer_row))


# ── 5. Rule trace ──────────────────────────────────────────────────────────────

@debug_bp.route('/rule-trace')
def rule_trace():
    """
    Trace a named dependency_rule binding-by-binding.  For each pair driven
    by the dep graph: temporal hazard check, bypass existence check,
    and violation decision are all reported separately.

    Query params: rule (str)  — exact rule name, e.g. 'RAW Hazard'
    Returns 404 if the rule is not registered.
    """
    rule_name = request.args.get('rule', '')
    result = _scheduler().evaluate_rules_trace(rule_name)
    if result is None:
        return jsonify({'error': f"Rule '{rule_name}' not found"}), 404
    return jsonify(result)


# ── 6. Raw scheduler state ─────────────────────────────────────────────────────

@debug_bp.route('/raw-state')
def raw_state():
    """Full current scheduler state — identical to GET /api/state."""
    return jsonify(_scheduler().get_state())