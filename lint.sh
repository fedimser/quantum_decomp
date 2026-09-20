result=0

echo "Running black..."
black --check ./quantum_decomp --fast
result+=$?

echo "Running pyright..."
pyright ./quantum_decomp
result+=$?

exit $result