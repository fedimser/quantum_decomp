from quantum_decomp.src.gate import GateSingle, GateFC
from quantum_decomp.src.gate2 import Gate2

def skip_identities(gates):
    return [gate for gate in gates if not gate.gate2.is_identity()]


def optimize_gates(gates):
    """Applies equivalent optimizations to sequence of gates."""
    return skip_identities(gates)
