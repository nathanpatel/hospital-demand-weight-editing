# Targeted Weight Editing for Hospital Demand Prediction

I was interested in Rikiya Takehi's article [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). It asks whether a language model can reason about another neural network and change its weights directly, rather than retraining the whole model.

I wanted to test a small version of that idea on a real prediction problem:

> Can an LLM change a few weights in an otherwise useful model to repair a specific mistake without creating worse mistakes elsewhere?

The task is deliberately modest. The model uses the previous twelve months of aggregate NHS England A&E attendances for one provider and predicts whether the following month will be unusually busy relative to that provider's own recent history. The model is a small 1D CNN with eight hidden channels and two output scores: ordinary demand and high demand.

## What I found

The provider-level CNN had held-out ROC-AUC 0.670, compared with 0.674 for logistic regression and 0.565 for seasonal persistence. It was useful enough to study, but not a production forecasting system.

I found a repeated weakness: the CNN often missed high-demand months after a flat or falling six-month lead-in. I then compared three ways of changing its decisions on a validation period:

| Method | Flat/falling recall | False alarms | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Original CNN | 49.5% | 249 | 64.4% |
| Threshold 0.5 | 72.0% | 377 | 62.2% |
| Analytical weight edit | 85.0% | 493 | 57.6% |
| LLM weight edit | 79.4% | 419 | 60.8% |
| Logistic regression | 69.2% | 252 | 68.9% |

Both weight edits improved recall on the target pattern, but both produced too many false alarms. The LLM made a valid, small patch, but it did not selectively repair the model's behaviour. That was the useful result: a better score on selected examples was not the same as a better model.

This is a research and learning project, not a hospital forecasting tool. The data are aggregate public observations, and the results do not have clinical or operational significance.

## Reproducing the experiment

The repository expects Python 3.11 and uses a local virtual environment:

\`\`\`bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
\`\`\`

Download the official NHS England provider-level monthly A&E CSV files for April 2023 to March 2026 from the [A&E Attendances and Emergency Admissions statistics page](https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/). Put the CSV files in \`data/raw/provider_monthly/\`. The raw data are intentionally ignored by Git.

The main commands are:

\`\`\`bash
.venv/bin/python provider_baseline.py --raw-directory data/raw/provider_monthly
.venv/bin/python diagnose_provider_errors.py --raw-directory data/raw/provider_monthly
.venv/bin/python analytical_provider_edit.py --raw-directory data/raw/provider_monthly
.venv/bin/python make_report_figures.py
\`\`\`

The LLM experiment is one controlled request and requires an \`OPENAI_API_KEY\` environment variable:

\`\`\`bash
export OPENAI_API_KEY=\"your-key-here\"
.venv/bin/python llm_provider_edit.py --raw-directory data/raw/provider_monthly
\`\`\`

The script validates the returned JSON patch, edits a copy of the checkpoint and records the result under \`outputs/\`. It does not modify the original checkpoint. Generated outputs, checkpoints, raw data, virtual environments and LaTeX build files are ignored.

The report source is [report/neural-weight-editing.tex](report/neural-weight-editing.tex). The main figures are in [figures/](figures/). A compiled PDF is not included because a LaTeX compiler was not available in the development environment.
