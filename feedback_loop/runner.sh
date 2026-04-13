FEEDBACK_NUM=5
FRAMEWORK="cirq"
MODELS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

PIPELINE_DIR="$REPO_ROOT/feedback_loop"
API_SCRIPT="$PIPELINE_DIR/api.py"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: could not find a Python interpreter (looked for $PYTHON_BIN and python)." >&2
    exit 1
  fi
fi

print_help () {
  echo "Usage: bash $(basename "$0") [--framework F] [--pass_k K] model_1 model_2 ..."
  echo ""
  echo "  --framework, --lang    One of: cirq | qiskit | pennylane   (default: cirq)"
  echo "  --feedback_num         Max attempts per task (default: 5)"
  echo ""
  echo "Examples:"
   echo "  bash $(basename "$0") --framework cirq --feedback_num 5 deepseek/deepseek-r1"
  echo "  bash $(basename "$0") --framework qiskit openai/gpt-4.1"
  echo "  bash $(basename "$0") --framework pennylane openai/gpt-4.1 deepseek/deepseek-r1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --feedback_num)
      FEEDBACK_NUM="$2"
      shift 2
      ;;
    --framework|--lang)
      FRAMEWORK="$2"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      MODELS+=("$1")
      shift
      ;;
  esac
done

if [ ${#MODELS[@]} -eq 0 ]; then
  print_help
  exit 1
fi

cd "$REPO_ROOT" || exit 1

echo "Configuration:"
echo "  Framework: $FRAMEWORK"
echo "  Feedback attempts: $FEEDBACK_NUM"
echo "  Models:"
for m in "${MODELS[@]}"; do
  echo "    - $m"
done
echo "---"

ARGS=( --framework "$FRAMEWORK" --feedback_num "$FEEDBACK_NUM")
"$PYTHON_BIN" "$API_SCRIPT" "${ARGS[@]}" "${MODELS[@]}"

echo "---"
echo "Feedback-loop run complete."
