using System;

using Microsoft.Quantum.Simulation.Core;
using Microsoft.Quantum.Simulation.Simulators;
using System.Numerics;
using System.Diagnostics;

namespace Quantum.QuantumDecompTest
{
    class Driver
    {
        private static int QUBITS_COUNT = 2;
        private static double EPS = 1e-9;

        static void Main(string[] args)
        {
            using (var qsim = new QuantumSimulator())
            {
                TestDump.Run(qsim, QUBITS_COUNT).Wait();
            }

            Complex[,] matrixUnderTest = readUnitary(2);

            double sqrt3 = Math.Sqrt(3);
            Complex w = Complex.Exp(2.0 / 3.0 * Math.PI * Complex.ImaginaryOne);
            Complex[,] referenceMatrix = new Complex[,] {
             {1.0/sqrt3, 1.0/sqrt3, 1.0/sqrt3, 0},
             {1.0/sqrt3, w/sqrt3, w*w/sqrt3, 0},
             {1.0/sqrt3, w*w/sqrt3, w/sqrt3, 0},
             {0, 0, 0, 1}};

            compareMatrices(matrixUnderTest, referenceMatrix);
            Console.WriteLine("OK");
            Console.ReadLine();
        }

        public static Complex[,] readUnitary(int N)
        {
            int size = 1 << N;
            Complex[,] unitary = new Complex[size, size];
            for (int column = 0; column < size; ++column)
            {
                string fileName = $"DU_{N}_{column}.txt";
                string[] fileContent = System.IO.File.ReadAllLines(fileName);
                for (int row = 0; row < size; ++row)
                {
                    string line = fileContent[row + 1];
                    string[] parts = line.Split('\t');
                    double real = Convert.ToDouble(parts[1]);
                    double imag = Convert.ToDouble(parts[2]);
                    if (Math.Abs(real) < EPS) real = 0;
                    if (Math.Abs(imag) < EPS) imag = 0;
                    unitary[column, row] = new Complex(real, imag);
                }
                System.IO.File.Delete(fileName);
            }
            return unitary;
        }

        // Compares that uniatry matrices are equal.
        // Allows for common phase shift for each row.
        public static void compareMatrices(Complex[,] matrix1, Complex[,] matrix2)
        {
            int N = matrix1.GetLength(0);

            for (int i = 0; i < N; ++i)
            {
                Complex allowedPhaseShift = 0;
                for (int j = 0; j < N; ++j)
                {
                    if (matrix1[i, j].Magnitude < EPS)
                    {
                        Debug.Assert(matrix2[i, j].Magnitude < EPS);
                    }
                    else
                    {
                        if (allowedPhaseShift == 0) {
                            allowedPhaseShift = matrix2[i, j] / matrix1[i, j];
                            Debug.Assert(Math.Abs(allowedPhaseShift.Magnitude-1)<EPS);
                        } 
                        Debug.Assert((matrix1[i, j] * allowedPhaseShift - matrix2[i, j]).Magnitude < EPS);
                    }
                }
            }
        }
    }
}