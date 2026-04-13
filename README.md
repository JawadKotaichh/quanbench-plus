# QuanBench Plus

LLM evaluation pipelines for quantum coding tasks across `cirq`, `qiskit`, and `pennylane`.

## 1) Setup

Run all commands from the repository root. Python `3.10+` is required.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

`pip install -e .` uses the project configuration from `pyproject.toml`.

## 2) Configure API key

Create `.env` in the repo root:

```env
API_KEY=your_openrouter_api_key
```

The pipelines call OpenRouter (`https://openrouter.ai/api/v1/chat/completions`) and read `API_KEY`.

## 3) Run Pass@k pipeline

### Bash runner only

```bash
bash pass_at_k_pipeline/runner.sh --framework cirq --pass_k 5 "openai/gpt-4.1"
bash pass_at_k_pipeline/runner.sh --framework qiskit --pass_k 5 "openai/gpt-4.1"
bash pass_at_k_pipeline/runner.sh --framework pennylane --pass_k 5 "openai/gpt-4.1"
```

## 4) Run Feedback-loop pipeline

```bash
bash feedback_loop/runner.sh --framework cirq --feedback_num 5 "openai/gpt-4.1"
bash feedback_loop/runner.sh --framework qiskit --feedback_num 5 "openai/gpt-4.1"
bash feedback_loop/runner.sh --framework pennylane --feedback_num 5 "openai/gpt-4.1"
```

## 5) Where results are saved

- Pass@k raw model outputs: `model_responses/<framework>/pass_at_*/*.json`
- Pass@k parsed responses: `responses/<framework>/pass_at_*/*.json`
- Pass@k evaluated results: `results/<framework>/pass_at_*/*.json`
- Feedback attempts + final state: `model_responses/<framework>/feedback_loop/*_attempts.json` and `*_final.json`

## Notes

- Model names must be OpenRouter IDs (example: `"openai/gpt-4.1"`).
- Runner scripts require at least one model argument.
- Use `--help` for CLI options:
  - `bash pass_at_k_pipeline/runner.sh --help`
  - `bash feedback_loop/runner.sh --help`
