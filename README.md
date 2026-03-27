# 6 Bandit Strategies (Streamlit Web App)

This project is a Streamlit web app for comparing 6 multi-armed bandit strategies in an explore-exploit setting.

## Problem Setup

- Total budget (default): `10000`
- Bandit means (default):
  - A: `0.8`
  - B: `0.7`
  - C: `0.5`
- Methods compared:
  - A/B Testing
  - Optimistic Initial Values
  - Epsilon-Greedy
  - Softmax (Boltzmann)
  - UCB
  - Thompson Sampling

## Files

- `streamlit_app.py`: main Streamlit app
- `requirements.txt`: Python dependencies

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the web app:

```bash
streamlit run streamlit_app.py
```

3. Open the shown URL in your browser (usually `http://localhost:8501`).

## App Features

- Interactive sidebar controls:
  - Budget, A/B exploration budget, Monte Carlo runs, random seed
  - True means of A/B/C
  - Hyperparameters for epsilon-greedy, softmax, UCB, and optimistic initialization
- Output summary table:
  - Expected reward
  - Regret
  - Average dollar allocation to A/B/C
- Performance charts:
  - Average return rate vs dollars spent
  - Expected reward and regret comparison

## Notes

- Monte Carlo runs can be increased for smoother curves (at the cost of runtime).
- Regret is computed against the optimal policy that always chooses the best true-mean arm.

## Suggested Git Commit

```bash
git add .
git commit -m "Add Streamlit app for 6 bandit strategy comparison"
```
