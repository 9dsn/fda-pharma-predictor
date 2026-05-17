# fda-pharma-predictor

An NLP system that predicts FDA Advisory Committee voting outcomes from briefing documents, and validates predictions as a leading indicator for drug approval decisions and biotech stock moves.

## Motivation
[1-2 paragraphs: why this problem matters, why AdComs specifically, what's novel here]

## Approach
- **Data**: FDA AdCom meeting records, briefing documents, ClinicalTrials.gov trial outcomes, historical approval data
- **Model**: [TBD]
- **Evaluation**: Time-based splits, calibration analysis, ablations

## Results


## Setup
\`\`\`bash
git clone https://github.com/9dsn/fda-pharma-predictor
cd fda-pharma-predictor
pip install -e .
cp .env.example .env  # add your API keys
\`\`\`

## Repo structure
[paste the tree from above]

## Methodology
See [docs/methodology.md](docs/methodology.md) for detailed reasoning behind data choices, modeling decisions, and evaluation approach.

## License
MIT
