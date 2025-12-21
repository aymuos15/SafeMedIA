#!/bin/bash
# Quick test script for DP-FedMed implementation

echo "=========================================="
echo "DP-FedMed Implementation Test"
echo "=========================================="
echo ""

# Check if in correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Not in project root directory"
    exit 1
fi

echo "1. Checking modified files..."
echo "   - dp_fedmed/privacy/budget_calculator.py"
[ -f "dp_fedmed/privacy/budget_calculator.py" ] && echo "     ✓ Found" || echo "     ✗ Missing"

echo "   - dp_fedmed/logging_config.py"
[ -f "dp_fedmed/logging_config.py" ] && echo "     ✓ Found" || echo "     ✗ Missing"

echo "   - dp_fedmed/server_app.py"
[ -f "dp_fedmed/server_app.py" ] && echo "     ✓ Found" || echo "     ✗ Missing"

echo "   - dp_fedmed/client_app.py"
[ -f "dp_fedmed/client_app.py" ] && echo "     ✓ Found" || echo "     ✗ Missing"

echo ""
echo "2. Checking old results directory..."
if [ -d "results" ]; then
    echo "   ✗ Old results/ directory still exists"
    echo "     Run: rm -rf results/"
else
    echo "   ✓ Old results/ directory removed"
fi

echo ""
echo "3. Configuration check..."
if [ -f "configs/default.toml" ]; then
    echo "   ✓ configs/default.toml exists"
    TARGET_EPSILON=$(grep "target_epsilon" configs/default.toml | head -1 | awk '{print $3}')
    echo "   Target epsilon: $TARGET_EPSILON"
else
    echo "   ✗ configs/default.toml not found"
fi

echo ""
echo "=========================================="
echo "Ready to run!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  bash scripts/run.sh"
echo ""
echo "Expected results directory structure:"
echo "  results/"
echo "  └── default/"
echo "      ├── server/"
echo "      │   ├── train.log"
echo "      │   ├── metrics.json"
echo "      │   └── history.json"
echo "      ├── client_0/"
echo "      │   ├── train.log"
echo "      │   ├── metrics.json"
echo "      │   └── history.json"
echo "      └── client_1/"
echo "          ├── train.log"
echo "          ├── metrics.json"
echo "          └── history.json"
echo ""
echo "After running, verify:"
echo "  1. Privacy budget calculation in server logs"
echo "  2. Final metrics are NOT 0.0 in server/metrics.json"
echo "  3. Cumulative epsilon ≤ target epsilon"
echo "  4. All clients have their own metrics.json"
echo ""
