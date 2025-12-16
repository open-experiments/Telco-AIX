#!/bin/bash
# Quick code validation script (no dependencies needed)

echo "======================================"
echo " AI-RAN Code Validation"
echo "======================================"
echo ""

cd "$(dirname "$0")"

echo "1. Checking Python syntax..."
ERROR=0

for file in $(find src -name "*.py" 2>/dev/null) quickstart.py test_installation.py; do
    if [ -f "$file" ] && python3 -m py_compile "$file" 2>/dev/null; then
        echo "  ✓ $file"
    elif [ -f "$file" ]; then
        echo "  ✗ $file FAILED"
        ERROR=1
    fi
done

echo ""
if [ $ERROR -eq 0 ]; then
    echo "✅ All Python files have valid syntax!"
    echo ""
    echo "Next steps:"
    echo "  1. Install dependencies: pip install -r requirements.txt"
    echo "  2. Run tests: python test_installation.py"
    echo "  3. Try demo: python quickstart.py"
    exit 0
else
    echo "❌ Some files have syntax errors. Please fix them."
    exit 1
fi
