#!/bin/bash
# Verify all four CLI tools are installed and accessible

echo "=== Checking CLI Tool Availability ==="
echo ""

check_tool() {
    local name="$1"
    local cmd="$2"
    if command -v "$cmd" &>/dev/null; then
        version=$($cmd --version 2>/dev/null | head -1)
        echo "✓ $name: $version"
    else
        echo "✗ $name: NOT FOUND ($cmd)"
    fi
}

check_tool "Claude Code" "claude"
check_tool "Codex CLI" "codex"
check_tool "Gemini CLI" "gemini"

check_tool "GitHub Copilot CLI" "copilot"

echo ""
echo "=== Checking Python Dependencies ==="
python -c "import mcp; print(f'✓ mcp: {mcp.__version__}')" 2>/dev/null || echo "✗ mcp: not installed"
python -c "import sentence_transformers; print(f'✓ sentence-transformers: {sentence_transformers.__version__}')" 2>/dev/null || echo "✗ sentence-transformers: not installed"
python -c "import faiss; print(f'✓ faiss: {faiss.__version__}')" 2>/dev/null || echo "✗ faiss: not installed"
python -c "import tree_sitter; print(f'✓ tree-sitter: {tree_sitter.__version__}')" 2>/dev/null || echo "✗ tree-sitter: not installed"
python -c "import tiktoken; print(f'✓ tiktoken: {tiktoken.__version__}')" 2>/dev/null || echo "✗ tiktoken: not installed"
python -c "import scipy; print(f'✓ scipy: {scipy.__version__}')" 2>/dev/null || echo "✗ scipy: not installed"
python -c "import matplotlib; print(f'✓ matplotlib: {matplotlib.__version__}')" 2>/dev/null || echo "✗ matplotlib: not installed"

echo ""
echo "=== Checking GPU ==="
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "✗ PyTorch not installed (needed for GPU)"
