namespace Quantum.QuantumDecompTest
{
    open Microsoft.Quantum.Primitive;
    open Microsoft.Quantum.Canon;
	open Microsoft.Quantum.Extensions.Convert;
	open Microsoft.Quantum.Extensions.Math;
	open Microsoft.Quantum.Extensions.Diagnostics;
    
    operation TestDump(N : Int) : Unit {
        DumpUnitary(N, ApplyUnitaryMatrix);
    }

	// Under test.
	operation ApplyUnitaryMatrix (qs : Qubit[]) : Unit {
		body (...) {
			Controlled Ry([qs[1]], (-3.141592653589793, qs[0]));
			Controlled R1([qs[1]], (3.141592653589793, qs[0]));
			Controlled Ry([qs[0]], (-1.570796326794897, qs[1]));
			Controlled R1([qs[0]], (3.141592653589793, qs[1]));
			X(qs[1]);
			Controlled Ry([qs[1]], (-1.910633236249018, qs[0]));
			Controlled R1([qs[1]], (3.141592653589793, qs[0]));
			X(qs[1]);
			Controlled Rz([qs[0]], (1.570796326794896, qs[1]));
			Controlled Ry([qs[0]], (-1.570796326794897, qs[1]));
			Controlled Rz([qs[0]], (-1.570796326794896, qs[1]));
			Controlled R1([qs[0]], (3.141592653589793, qs[1]));
			Controlled Rz([qs[1]], (1.570796326794897, qs[0]));
			Controlled Ry([qs[1]], (-3.141592653589793, qs[0]));
			Controlled Rz([qs[1]], (-1.570796326794897, qs[0]));
			Controlled R1([qs[1]], (-1.570796326794897, qs[0]));
		}
	}

	// Applies unitary operation to every basis state and dumps results to files.
	operation DumpUnitary (N : Int, unitary: (Qubit[] => Unit)) : Unit {
		body (...) {
			let size = 1 <<< N;
        
			using (qs = Qubit[N]) {
				for (k in 0 .. size - 1) {                
					let binary = BoolArrFromPositiveInt(k, N);
					ApplyPauliFromBitString(PauliX, true, binary, qs);
					unitary(qs);
					DumpMachine($"DU_{N}_{k}.txt");
					ResetAll(qs);
				}
			}
        }
    }

}
