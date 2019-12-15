from src.gate import GateSingle, GateFC
from src.gate2 import Gate2


def rearrange_for_merge(gates):
    """Rearranges gates, so mergeable gates are next to each other.

    Swaps only 1-qubit gates acting on different qubits, which always commute.
    """
    n = len(gates)
    result = []
    taken = [False] * n
    pos = 0
    while(pos != n):
        if taken[pos]:
            pos += 1
        elif isinstance(gates[pos], GateFC):
            result.append(gates[pos])
            taken[pos] = True
            pos += 1
        else:
            assert isinstance(gates[pos], GateSingle)
            # Take first gate with minimal qubit_id.
            to_take = pos
            for i in range(pos + 1, n):
                if taken[i]:
                    continue
                if isinstance(gates[i], GateFC):
                    break
                assert isinstance(gates[i], GateSingle)
                if gates[i].qubit_id < gates[to_take].qubit_id:
                    to_take = i
            result.append(gates[to_take])
            taken[to_take] = True

    assert len(result) == n
    return result


def merge_same_gates(gates):
    def can_merge(g1, g2):
        """Checks if gates can be merged."""
        assert (g1.qubit_count == g2.qubit_count)
        if isinstance(g1, GateSingle) and isinstance(g2, GateSingle):
            if (g1.qubit_id != g2.qubit_id):
                return False
        else:
            return False
        if g1.gate2.name not in ['Rx', 'Ry', 'Rz', 'R1']:
            return False
        return g1.gate2.name == g2.gate2.name

    def merge_gates(g1, g2):
        assert can_merge(g1, g2)
        new_gate2 = Gate2(g1.gate2.name, arg=g1.gate2.arg + g2.gate2.arg)
        return GateSingle(new_gate2, g1.qubit_id, g1.qubit_count)

    if len(gates) == 0:
        return []

    result = []
    cur_gate = gates[0]
    for gate in gates[1:]:
        if can_merge(cur_gate, gate):
            cur_gate = merge_gates(cur_gate, gate)
        else:
            result.append(cur_gate)
            cur_gate = gate
    result.append(cur_gate)
    return result


def skip_identities(gates):
    return [gate for gate in gates if not gate.gate2.is_identity()]


def optimize_gates(gates):
    """
    Applies equivalent optimizations to sequence of gates.

    Merges adjacent Rx, Ry, Rz and R1 gates.
    Cancels consequent NOT gates.
    Skips identity gates.

    After execution all fully controlled gates will have `flip_mask=0`.
    """
    qubit_count = gates[0].qubit_count
    for gate in gates:
        assert gate.qubit_count == qubit_count

    result = []

    global flip_mask
    flip_mask = 0

    def dump_flips():
        global flip_mask
        for qubit_id in range(qubit_count):
            if (flip_mask & (2**qubit_id)) != 0:
                result.append(GateSingle(Gate2('X'), qubit_id, qubit_count))
        flip_mask = 0

    for gate in gates:
        if isinstance(gate, GateSingle):
            if gate.gate2.name == 'X':
                flip_mask ^= 2**gate.qubit_id
            elif gate.gate2.is_identity():
                pass
            else:
                dump_flips()
                result.append(gate)
        else:
            assert isinstance(gate, GateFC)
            if gate.gate2.is_identity():
                pass
            else:
                flip_mask ^= gate.flip_mask
                dump_flips()
                result.append(gate.without_flips())
                flip_mask ^= gate.flip_mask
    dump_flips()

    result = rearrange_for_merge(result)
    result = merge_same_gates(result)
    result = skip_identities(result)

    return result
