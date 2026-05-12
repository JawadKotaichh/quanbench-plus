PASS_K=1
FRAMEWORK="cirq"
BENCHMARK_VERSION="v1"
MODELS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

PIPELINE_DIR="$REPO_ROOT/pass_at_k_pipeline"
API_MODULE="pass_at_k_pipeline.api_pass_at_k"

# Sanity: python exists
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
  echo "  --pass_k               Pass@k samples (default: 1)"
  echo "  --benchmark-version    One of: v1 | v2 (default: v1)"
  echo ""
  echo "Examples:"
  echo "  bash $(basename "$0") --framework cirq --pass_k 5 "openai/gpt-4.1""
  echo "  bash $(basename "$0") --framework qiskit "deepseek/deepseek-chat""
  echo "  bash $(basename "$0") --framework pennylane "openai/gpt-4.1" "deepseek/deepseek-r1""
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pass_k)
      PASS_K="$2"
      shift 2
      ;;
    --framework|--lang)
      FRAMEWORK="$2"
      shift 2
      ;;
    --benchmark-version)
      BENCHMARK_VERSION="$2"
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


case "$FRAMEWORK" in
  cirq)
    RESULTS_MODULE="pass_at_k_pipeline.cirq_pip.get_cirq_results"
    ;;
  qiskit)
    RESULTS_MODULE="pass_at_k_pipeline.qiskit_pip.get_qiskit_results"
    ;;
  pennylane)
    RESULTS_MODULE="pass_at_k_pipeline.pennylane_pip.get_pennylane_results"
    ;;
  *)
    echo "Error: unknown framework '$FRAMEWORK'. Use: cirq | qiskit | pennylane" >&2
    exit 1
    ;;
esac

cd "$REPO_ROOT" || exit 1

echo "Configuration:"
echo "  Framework: $FRAMEWORK"
echo "  Pass@k samples: $PASS_K"
echo "  Benchmark version: $BENCHMARK_VERSION"
echo "  Models:"
for m in "${MODELS[@]}"; do
  echo "    - $m"
done
echo "---"

"$PYTHON_BIN" -m "$API_MODULE" --framework "$FRAMEWORK" --pass_k "$PASS_K" --benchmark-version "$BENCHMARK_VERSION" "${MODELS[@]}"
"$PYTHON_BIN" -m "$RESULTS_MODULE" --benchmark-version "$BENCHMARK_VERSION" "${MODELS[@]}" "$PASS_K"

echo "---"
echo "All evaluations complete."
