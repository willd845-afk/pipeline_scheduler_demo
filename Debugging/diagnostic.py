#!/usr/bin/env python3
"""
Pipeline Scheduler Diagnostic Script (v3 — Self-Contained)

Sets up a known two-instruction RAW scenario via the API, runs full
diagnostics, tests that a valid bypass resolves the violation, then
restores the original scheduler state.

Usage:
    python diagnostic.py          # run and restore original state
    python diagnostic.py --keep   # leave test state in the scheduler
"""

import sys
import json
import requests
import time
import traceback
from typing import Any, Dict, List, Optional

# ─── Configuration ────────────────────────────────────────────────────────────
BASE_URL     = "http://localhost:5000"
DEBUG_BASE   = f"{BASE_URL}/api/debug"
PRODUCER_ROW = 0
CONSUMER_ROW = 1

# ─── Test scenario ────────────────────────────────────────────────────────────
# Producer: add r1, r2, r3  (writes r1)
# Consumer: add r4, r1, r5  (reads r1  → RAW hazard on r1)
#
# Stage layout (skipping i* stall stage for clarity):
#
#  Row 0 (producer):  F=0  D=1  I=2  X=3  W=4  r*=5  C=6
#  Row 1 (consumer):  F=1  D=2  I=3  X=4  W=5  r*=6  C=7
#
# Hazard:  col(C of row0)=6  >=  col(X of row1)=4   → RAW VIOLATION expected
#
# Valid bypass window:
#   source  must be >= col(X of row0) = 3        (when_done  = X)
#   target  must be >= col(D of row1) = 2        (decoded    = D)
#   target  must be <  col(X of row1) = 4        (when_start = X)
#   → (row0,col3=X) → (row1,col2=D)  is VALID
#   → (row0,col3=X) → (row1,col4=X)  is INVALID  (too late)
#   → (row0,col1=D) → (row1,col2=D)  is INVALID  (too early)

PRODUCER_BLOCKS: Dict[int, str] = {
    0: 'F', 1: 'D', 2: 'I', 3: 'X', 4: 'W', 5: 'r*', 6: 'C'
}
CONSUMER_BLOCKS: Dict[int, str] = {
    1: 'F', 2: 'D', 3: 'I', 4: 'X', 5: 'W', 6: 'r*', 7: 'C'
}
PRODUCER_INSTR = {
    'name': 'add',
    'operands': {'rd': 'r1', 'rs1': 'r2', 'rs2': 'r3'},
}
CONSUMER_INSTR = {
    'name': 'add',
    'operands': {'rd': 'r4', 'rs1': 'r1', 'rs2': 'r5'},
}

VALID_BYPASS   = {'source': {'row': 0, 'col': 3},   # X stage of producer
                  'target': {'row': 1, 'col': 2}}    # D stage of consumer

INVALID_BYPASS_LATE   = {'source': {'row': 0, 'col': 3},
                          'target': {'row': 1, 'col': 4}}  # lands on X itself

INVALID_BYPASS_EARLY  = {'source': {'row': 0, 'col': 1},
                          'target': {'row': 1, 'col': 2}}  # source before X


# ─── HTTP helpers ─────────────────────────────────────────────────────────────

def _get(path: str, params: Dict = None) -> Optional[Dict]:
    try:
        r = requests.get(f"{BASE_URL}{path}", params=params, timeout=5)
        if r.status_code == 404:
            print(f"  ❌ 404 {path} — blueprint not registered. "
                  f"Check FLASK_DEBUG=1 and init_debug_blueprint().")
            return None
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        print(f"  ❌ Cannot reach {BASE_URL}. Is the server running?")
        sys.exit(1)
    except Exception as e:
        print(f"  ❌ GET {path}: {e}")
        return None


def _post(path: str, body: Dict) -> Optional[Dict]:
    try:
        r = requests.post(f"{BASE_URL}{path}", json=body,
                          headers={'Content-Type': 'application/json'}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ❌ POST {path}: {e}")
        return None


def _delete(path: str, body: Dict) -> Optional[Dict]:
    try:
        r = requests.delete(f"{BASE_URL}{path}", json=body,
                            headers={'Content-Type': 'application/json'}, timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  ❌ DELETE {path}: {e}")
        return None


# ─── State management ─────────────────────────────────────────────────────────

def print_header(text):
    """Prints a major section header for the diagnostic."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_subheader(text):
    """Prints a minor section header."""
    print(f"\n── {text} " + "─" * (56 - len(text)))

def print_result(label, success):
    """Prints a consistent success/fail indicator."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"  {status}  {label}")

def save_state() -> Dict:
    """Save the current scheduler state so we can restore it later."""
    data = _get('/api/state')
    return data or {}


def restore_state(state: Dict) -> None:
    """Restore a previously saved state."""
    _post('/api/state', state)
    print("  ↩  Original state restored.")


def clear_state() -> None:
    """Clear all blocks, instructions, and bypass annotations."""
    _post('/api/state', {
        'grid_data': {},
        'instructions': {},
        'rows': 10,
        'cols': 10,
        'bypass_annotations': [],
    })


def setup_test_scenario() -> None:
    """
    Install a known two-instruction RAW scenario into the scheduler.
    """
    clear_state()

    # Place blocks
    for col, stage in PRODUCER_BLOCKS.items():
        _post('/api/block', {'row': PRODUCER_ROW, 'col': col, 'block_type': stage})
    for col, stage in CONSUMER_BLOCKS.items():
        _post('/api/block', {'row': CONSUMER_ROW, 'col': col, 'block_type': stage})

    # Assign instructions
    _post('/api/instruction', {'row': PRODUCER_ROW, 'instruction': PRODUCER_INSTR})
    _post('/api/instruction', {'row': CONSUMER_ROW, 'instruction': CONSUMER_INSTR})

    print("  ✅ Test scenario installed:")
    print(f"     Row {PRODUCER_ROW}: add r1, r2, r3  "
          f"(producer — writes r1) | stages: {PRODUCER_BLOCKS}")
    print(f"     Row {CONSUMER_ROW}: add r4, r1, r5  "
          f"(consumer — reads  r1) | stages: {CONSUMER_BLOCKS}")


def add_bypass(source: Dict, target: Dict) -> bool:
    """Add a bypass annotation. Returns True on success."""
    data = _post('/api/bypass-annotations', {
        'annotation_type': 'RAW',
        'source': source,
        'target': target,
    })
    return bool(data and data.get('success'))


def clear_all_bypasses() -> None:
    """Remove all bypass annotations."""
    state = _get('/api/state')
    if not state:
        return
    for ann in state.get('bypass_annotations', []):
        _delete('/api/bypass-annotations', {
            'source':          ann['source'],
            'target':          ann['target'],
            'annotation_type': ann['annotation_type'],
        })


# ─── Diagnostic sections ──────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def check_instruction_formats() -> bool:
    section("1 · Instruction Formats")
    data = _get('/api/debug/instruction-formats')
    if not data:
        return False

    REQUIRED = ['when_start', 'when_done', 'values_committed',
                'decoded', 'read_dependency', 'write_dependency']

    ok = True
    for name, fmt in data.items():
        missing = [f for f in REQUIRED if not fmt.get(f)]
        if missing:
            print(f"  ❌ {name}: missing/empty fields: {missing}")
            ok = False
        else:
            print(f"  ✅ {name}:")
            for f in REQUIRED:
                print(f"       {f:<20} = {fmt[f]}")

    return ok


def check_stage_columns() -> bool:
    section("2 · Stage Columns (verifying test scenario is loaded)")
    data = _get('/api/debug/stage-columns')
    if not data:
        return False

    if not data:
        print("  ❌ Empty response — no instructions in scheduler.")
        print("     This means init_debug_blueprint() is accessing a different")
        print("     scheduler instance than the one modified by /api/block.")
        print("     Verify Fix 1 and Fix 2 were applied to app.py.")
        return False

    ok = True
    REQUIRED_ATTRS = ['when_start', 'when_done', 'values_committed', 'decoded']

    for row_str, info in data.items():
        print(f"\n  Row {row_str} ({info['instruction']}):")
        stages = info.get('stages', {})

        for attr in REQUIRED_ATTRS:
            if attr not in stages:
                print(f"    ❌ {attr:<20} — attribute not in response")
                ok = False
                continue
            s = stages[attr]
            col    = s['column']
            status = s['status']
            symbol = '✅' if status == 'ok' else ('⚠️ ' if status == 'not_placed' else '❌')
            print(f"    {symbol} {attr:<20} stage={s['stage_name']:<4} col={col}")
            if status == 'not_placed':
                print(f"       ↳ Stage '{s['stage_name']}' is declared but not placed in row {row_str}")
                ok = False
            elif status == 'not_declared':
                print(f"       ↳ Attribute not declared in InstructionFormat — check YAML")
                ok = False

    return ok


def _graph() -> bool:
    section("3 · Dependency Graph (verifying RAW edge exists)")
    data = _get('/api/debug/dep-graph-detail')
    if not data:
        return False

    summary = data.get('summary', {})
    print(f"  Total edges: {summary.get('total_edges', 0)}")
    print(f"  By class:    {summary.get('by_class', {})}")
    mismatches = summary.get('type_mismatches', 0)
    if mismatches:
        print(f"  ❌ {mismatches} type mismatch(es) — write_dep != read_dep")
        print("     This silently suppresses RAW edges.")

    raw_edges = [e for e in data.get('edges', []) if e['dep_class'] == 'RAW']
    if not raw_edges:
        print("\n  ❌ No RAW edges found.")
        print("     Check that BOTH instructions have the same dep type string.")

        nodes = data.get('nodes', {})
        for row_str, node in nodes.items():
            print(f"\n  Row {row_str} ({node['instruction_name']}):")
            print(f"    write_dep_type = '{node['write_dep_type']}'")
            print(f"    read_dep_type  = '{node['read_dep_type']}'")
            print(f"    written_regs   = {node['written_regs']}")
            print(f"    read_regs      = {node['read_regs']}")
        return False

    print(f"\n  ✅ Found {len(raw_edges)} RAW edge(s):")
    for e in raw_edges:
        print(f"     Row {e['producer_row']} → Row {e['consumer_row']} "
              f"via {e['shared_regs']}  (type_match={e['type_match']})")

    return True


def check_bypass_validation(
        source: Dict, target: Dict, expect_valid: bool, label: str
) -> bool:
    """
    Add a specific bypass, run bypass-validation, report result, then remove it.
    """
    print(f"\n  Testing: {label}")
    print(f"    source = row{source['row']},col{source['col']}  "
          f"target = row{target['row']},col{target['col']}")

    add_bypass(source, target)

    data = _get('/api/debug/bypass-validation',
                params={'producer_row': PRODUCER_ROW, 'consumer_row': CONSUMER_ROW})

    if not data:
        clear_all_bypasses()
        return False

    bounds = data.get('bounds', {})
    print(f"    Bounds: source >= {bounds.get('source_min_col')} ({bounds.get('source_stage')}), "
          f"target >= {bounds.get('target_min_col')} ({bounds.get('target_min_stage')}), "
          f"target <  {bounds.get('target_max_col')} ({bounds.get('target_max_stage')})")

    candidates = data.get('candidates', [])
    if not candidates:
        print(f"    ❌ Annotation not found in candidates list.")
        clear_all_bypasses()
        return False

    for c in candidates:
        src_ok = "✅" if c['check_source_gte_min'] else "❌"
        tgm_ok = "✅" if c['check_target_gte_min'] else "❌"
        tgx_ok = "✅" if c['check_target_lt_max']  else "❌"
        valid  = c['valid']
        reason = c.get('failure_reason') or '—'
        print(f"    {src_ok} src>=min  {tgm_ok} tgt>=min  {tgx_ok} tgt<max  "
              f"→ {'VALID' if valid else 'REJECTED: ' + reason}")

    actual_valid = data.get('any_valid', False)
    clear_all_bypasses()

    if actual_valid == expect_valid:
        symbol = "✅" if expect_valid else "✅"
        print(f"    {symbol} Correct: bypass {'accepted' if actual_valid else 'rejected'} as expected")
        return True
    else:
        symbol = "❌"
        print(f"    {symbol} Wrong: expected {'valid' if expect_valid else 'invalid'}, "
              f"got {'valid' if actual_valid else 'invalid'}")
        return False


def check_bypass_section() -> bool:
    section("4 · Bypass Column Bound Checks")
    results = [
        check_bypass_validation(
            VALID_BYPASS['source'], VALID_BYPASS['target'],
            expect_valid=True,
            label="VALID:   X(row0) → D(row1)  [should be accepted]",
        ),
        check_bypass_validation(
            INVALID_BYPASS_LATE['source'], INVALID_BYPASS_LATE['target'],
            expect_valid=False,
            label="INVALID: X(row0) → X(row1)  [lands on execution, should be rejected]",
        ),
        check_bypass_validation(
            INVALID_BYPASS_EARLY['source'], INVALID_BYPASS_EARLY['target'],
            expect_valid=False,
            label="INVALID: D(row0) → D(row1)  [source before when_done, should be rejected]",
        ),
    ]
    return all(results)


def check_rule_trace_no_bypass() -> bool:
    section("5 · Rule Trace: NO bypass (violation expected)")
    data = _get('/api/debug/rule-trace', params={'rule': 'RAW Hazard'})
    if not data:
        return False

    print(f"  Pairs evaluated: {data['bindings_evaluated']}")
    print(f"  Violations:      {data['violations_emitted']}")

    for t in data['trace']:
        th  = t['temporal_hazard']
        byp = t['bypass_exists']
        print(f"\n  Pair: Row {t['producer_row']} → Row {t['consumer_row']}  "
              f"shared={t['shared_regs']}")
        print(f"    temporal_hazard:  {th['producer_stage']}(col{th['producer_col']}) "
              f">= {th['consumer_stage']}(col{th['consumer_col']})  → {th['result']}")
        print(f"    bypass_resolved:  {byp['any_valid']}")
        print(f"    violation_emitted: {t['violation_emitted']}")

    if data['violations_emitted'] == 0:
        print("\n  ❌ Expected a violation but none was emitted.")
        if data['bindings_evaluated'] == 0:
            print("     0 pairs evaluated — dep graph has no RAW edges.")
            print("     Fix: verify dep_graph section (Section 3) first.")
        return False

    print("\n  ✅ Violation emitted as expected.")
    return True


def check_rule_trace_with_bypass() -> bool:
    section("6 · Rule Trace: WITH valid bypass (no violation expected)")

    ok = add_bypass(VALID_BYPASS['source'], VALID_BYPASS['target'])
    if not ok:
        print("  ❌ Could not add bypass annotation.")
        return False

    print(f"  Bypass added: row{VALID_BYPASS['source']['row']},"
          f"col{VALID_BYPASS['source']['col']} → "
          f"row{VALID_BYPASS['target']['row']},"
          f"col{VALID_BYPASS['target']['col']}")

    data = _get('/api/debug/rule-trace', params={'rule': 'RAW Hazard'})
    clear_all_bypasses()

    if not data:
        return False

    print(f"  Pairs evaluated: {data['bindings_evaluated']}")
    print(f"  Violations:      {data['violations_emitted']}")

    for t in data['trace']:
        byp = t['bypass_exists']
        print(f"\n  Pair: Row {t['producer_row']} → Row {t['consumer_row']}")
        print(f"    bypass_resolved:   {byp['any_valid']}")
        print(f"    violation_emitted: {t['violation_emitted']}")

        if not byp['any_valid']:
            print(f"\n    ↳ Bypass still not recognised. Per-annotation breakdown:")
            for ann in byp.get('annotations_on_pair', []):
                print(f"       {ann['source']} → {ann['target']}")
                print(f"         src >= source_min: {ann['passes_source_bound']}")
                print(f"         tgt >= target_min: {ann['passes_target_min']}")
                print(f"         tgt <  target_max: {ann['passes_target_max']}")

    if data['violations_emitted'] > 0:
        print("\n  ❌ Violation still emitted despite valid bypass.")
        print("     See breakdown above for which column check failed.")
        return False

    print("\n  ✅ Bypass resolved the violation correctly.")
    return True


def check_dep_graph() -> bool:
    """Verifies the dependency graph contains the expected RAW edge."""
    print_header("3 · Dependency Graph (verifying RAW edge exists)")

    data = _get('/api/debug/dep-graph-detail')
    if not data:
        return False

    nodes   = data.get('nodes', {})
    edges   = data.get('edges', [])
    summary = data.get('summary', {})

    print(f"  Nodes found:     {len(nodes)}")
    print(f"  Edges found:     {len(edges)}")
    print(f"  By class:        {summary.get('by_class', {})}")

    mismatches = summary.get('type_mismatches', 0)
    if mismatches:
        print(f"  ⚠️  {mismatches} dep_type mismatch(es) — "
              f"potential edges silently suppressed")

    raw_edge = next(
        (e for e in edges
         if e.get('producer_row') == PRODUCER_ROW
         and e.get('consumer_row') == CONSUMER_ROW
         and e.get('dep_class') == 'RAW'),
        None,
    )

    if raw_edge:
        regs = raw_edge.get('shared_regs', [])
        print(f"  ✅ RAW edge: Row {PRODUCER_ROW} → Row {CONSUMER_ROW} "
              f"(shared registers: {regs})")
        return True

    print(f"  ❌ No RAW edge found between Row {PRODUCER_ROW} "
          f"and Row {CONSUMER_ROW}.")
    print("     All edges present:")
    for e in edges:
        print(f"       {e.get('dep_class')} "
              f"Row {e.get('producer_row')} → Row {e.get('consumer_row')} "
              f"via {e.get('shared_regs')}")
    return False

# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    keep = '--keep' in sys.argv

    print("Pipeline Scheduler Diagnostic (v3 — Self-Contained)")
    print("=" * 60)
    print(f"  Server:       {BASE_URL}")
    print(f"  Producer row: {PRODUCER_ROW}  ({PRODUCER_INSTR['name']} "
          f"{PRODUCER_INSTR['operands']})")
    print(f"  Consumer row: {CONSUMER_ROW}  ({CONSUMER_INSTR['name']} "
          f"{CONSUMER_INSTR['operands']})")

    # ── Save original state ───────────────────────────────────────────
    print("\n  Saving current scheduler state...")
    original_state = save_state()

    # ── Install test scenario ─────────────────────────────────────────
    print("  Installing test scenario...")
    setup_test_scenario()

    # ── Run diagnostics ───────────────────────────────────────────────
    results = {
        'instruction_formats': check_instruction_formats(),
        'stage_columns':       check_stage_columns(),
        'dep_graph':           check_dep_graph(),
        'bypass_bounds':       check_bypass_section(),
        'rule_no_bypass':      check_rule_trace_no_bypass(),
        'rule_with_bypass':    check_rule_trace_with_bypass(),
    }

    # ── Summary ───────────────────────────────────────────────────────
    section("Summary")
    all_passed = True
    for name, passed in results.items():
        icon = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {icon}  {name}")
        if not passed:
            all_passed = False

    # ── Restore state ─────────────────────────────────────────────────
    if not keep:
        print()
        restore_state(original_state)
    else:
        print("\n  --keep flag set: test state left in scheduler.")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())