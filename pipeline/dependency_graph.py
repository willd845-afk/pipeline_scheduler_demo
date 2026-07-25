"""
pipeline/dependency_graph.py

Dynamic register dependency graph.

Nodes represent instructions assigned to grid rows.  Directed edges capture
three dependency classes for each ordered (producer, consumer) pair where
producer.row < consumer.row:

    RAW  write_dependency(producer) == read_dependency(consumer)
         AND writes(producer) ∩ reads(consumer)  ≠ ∅

    WAW  write_dependency(producer) == write_dependency(consumer)
         AND writes(producer) ∩ writes(consumer) ≠ ∅

    WAR  read_dependency(producer)  == write_dependency(consumer)
         AND reads(producer)  ∩ writes(consumer) ≠ ∅

Parent / child semantics (RAW only):
    parent(j) = { i : i writes what j reads }
    child(i)  = { j : j reads what i writes }

Call build() after any instruction name or operand change.  The graph is
consumed by QuantifiedYAMLRule (for dependency_rule iteration) and exposed
through GET /api/graphs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

from .instructions import _get_written_reg_values, _get_read_reg_values


@dataclass
class DepNode:
    """
    One instruction in the dependency graph.

    Attributes:
        row              (int):       Grid row index.
        instruction_name (str):       Instruction mnemonic.
        write_dep_type   (str):       Category written by this instruction.
        read_dep_type    (str):       Category read by this instruction.
        when_start       (str):       Stage where execution begins.
        when_done        (str):       Stage where result is available for bypass.
        values_committed (str):       Stage where result is committed to
                                       the register file.
        decoded          (str):       Stage where the instruction is decoded;
                                       earliest valid bypass target stage.
        written_regs     (Set[str]):  Concrete register values written.
        read_regs        (Set[str]):  Concrete register values read.
        parents          (List[int]): RAW predecessors (write what this reads).
        children         (List[int]): RAW successors  (read what this writes).
    """
    row:              int
    instruction_name: str
    write_dep_type:   str
    read_dep_type:    str
    when_start:       str
    when_done:        str
    values_committed: str
    decoded:          str
    written_regs:     Set[str]
    read_regs:        Set[str]
    parents:          List[int] = field(default_factory=list)
    children:         List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'row':              self.row,
            'instruction_name': self.instruction_name,
            'write_dep_type':   self.write_dep_type,
            'read_dep_type':    self.read_dep_type,
            'when_start':       self.when_start,
            'when_done':        self.when_done,
            'values_committed': self.values_committed,
            'decoded':          self.decoded,
            'written_regs':     sorted(self.written_regs),
            'read_regs':        sorted(self.read_regs),
            'parents':          sorted(self.parents),
            'children':         sorted(self.children),
        }


@dataclass
class DepEdge:
    """
    A directed dependency edge from producer_row to consumer_row.

    Attributes:
        producer_row  (int):      Row that produces (or reads, for WAR) the value.
        consumer_row  (int):      Row that consumes (or writes, for WAR) the value.
        dep_class     (str):      'RAW', 'WAW', or 'WAR'.
        dep_type      (str):      Dependency category e.g. 'reg_update'.
        shared_regs   (Set[str]): Registers that create the dependency.
    """
    producer_row: int
    consumer_row: int
    dep_class:    str
    dep_type:     str
    shared_regs:  Set[str]

    def to_dict(self) -> Dict:
        return {
            'producer_row': self.producer_row,
            'consumer_row': self.consumer_row,
            'dep_class':    self.dep_class,
            'dep_type':     self.dep_type,
            'shared_regs':  sorted(self.shared_regs),
        }


class DependencyGraph:
    """
    Dynamic register dependency graph over the current instruction set.

    Call build() whenever instructions or operand values change.  The graph
    is then consumed by QuantifiedYAMLRule (for filtered iteration per
    dep_class) and serialised for GET /api/graphs.
    """

    def __init__(self) -> None:
        self.nodes: Dict[int, DepNode] = {}
        self.edges: List[DepEdge]      = []

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, instructions: Dict[int, Any]) -> None:
        """
        Rebuild all nodes and edges from the current instruction assignments.

        Clears the existing graph before rebuilding so stale data from
        deleted or replaced instructions is not retained.

        Args:
            instructions: Maps row index to Instruction instance.
        """
        self.nodes.clear()
        self.edges.clear()

        # ── 1. Build nodes ──────────────────────────────────────────────
        for row, instr in instructions.items():
            if not instr or not getattr(instr, 'name', ''):
                continue
            fmt = getattr(instr, 'format', None)
            # May just be compliment of not fmt. Otherwise, you can do 'not fmt: continue'
            if fmt:
                self.nodes[row] = DepNode(
                    row=row,
                    instruction_name=instr.name,
                    write_dep_type=getattr(fmt, 'write_dependency', ''),
                    read_dep_type=getattr(fmt, 'read_dependency', ''),
                    when_start=getattr(fmt, 'when_start', ''),
                    when_done=getattr(fmt, 'when_done', ''),
                    values_committed=getattr(fmt, 'values_committed', ''),
                    decoded=getattr(fmt, 'decoded', ''),
                    written_regs=_get_written_reg_values(instr),
                    read_regs=_get_read_reg_values(instr),
                )



        # ── 2. Build edges for every ordered (producer, consumer) pair ──
        sorted_rows = sorted(self.nodes.keys())
        for p_idx, prod_row in enumerate(sorted_rows):
            producer = self.nodes[prod_row]
            for cons_row in sorted_rows[p_idx + 1:]:
                self._add_edges(producer, self.nodes[cons_row])

        # ── 3. Populate RAW parent / child links ─────────────────────────
        for edge in self.edges:
            if edge.dep_class != 'RAW':
                continue
            prod_node = self.nodes.get(edge.producer_row)
            cons_node = self.nodes.get(edge.consumer_row)
            if prod_node and edge.consumer_row not in prod_node.children:
                prod_node.children.append(edge.consumer_row)
            if cons_node and edge.producer_row not in cons_node.parents:
                cons_node.parents.append(edge.producer_row)

    def _add_edges(self, producer: DepNode, consumer: DepNode) -> None:
        """Check all three dependency classes and append matching edges."""

        # ── RAW ────────────────────────────────────────────────────────
        # write_dependency(producer) == read_dependency(consumer)
        if (producer.write_dep_type
                and producer.write_dep_type == consumer.read_dep_type):
            shared = producer.written_regs & consumer.read_regs
            if shared:
                self.edges.append(DepEdge(
                    producer_row=producer.row,
                    consumer_row=consumer.row,
                    dep_class='RAW',
                    dep_type=producer.write_dep_type,
                    shared_regs=shared,
                ))

        # ── WAW ────────────────────────────────────────────────────────
        # write_dependency(producer) == write_dependency(consumer)
        if (producer.write_dep_type
                and producer.write_dep_type == consumer.write_dep_type):
            shared = producer.written_regs & consumer.written_regs
            if shared:
                self.edges.append(DepEdge(
                    producer_row=producer.row,
                    consumer_row=consumer.row,
                    dep_class='WAW',
                    dep_type=producer.write_dep_type,
                    shared_regs=shared,
                ))

        # ── WAR ────────────────────────────────────────────────────────
        # read_dependency(producer) == write_dependency(consumer)
        if (producer.read_dep_type
                and producer.read_dep_type == consumer.write_dep_type):
            shared = producer.read_regs & consumer.written_regs
            if shared:
                self.edges.append(DepEdge(
                    producer_row=producer.row,
                    consumer_row=consumer.row,
                    dep_class='WAR',
                    dep_type=producer.read_dep_type,
                    shared_regs=shared,
                ))

    # ── Queries ────────────────────────────────────────────────────────────

    def get_edges_by_class(self, dep_class: str) -> List[DepEdge]:
        """Return all edges whose dep_class matches the given string."""
        return [e for e in self.edges if e.dep_class == dep_class]

    def get_edges_for_pair(
            self, producer_row: int, consumer_row: int,
    ) -> List[DepEdge]:
        """Return all edges between a specific ordered pair."""
        return [
            e for e in self.edges
            if e.producer_row == producer_row
            and e.consumer_row == consumer_row
        ]

    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    # ── Serialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            'nodes': {str(k): v.to_dict() for k, v in self.nodes.items()},
            'edges': [e.to_dict() for e in self.edges],
        }