from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


METHODS = [
    "ab_test",
    "optimistic",
    "epsilon_greedy",
    "softmax",
    "ucb",
    "thompson",
]

METHOD_LABELS = {
    "ab_test": "A/B Testing",
    "optimistic": "Optimistic Initial Values",
    "epsilon_greedy": "Epsilon-Greedy",
    "softmax": "Softmax (Boltzmann)",
    "ucb": "UCB",
    "thompson": "Thompson Sampling",
}


@dataclass
class SimulationConfig:
    mean_a: float
    mean_b: float
    mean_c: float
    total_budget: int
    explore_budget: int
    runs: int
    seed: int
    epsilon: float
    tau0: float
    tau_min: float
    tau_decay: float
    ucb_c: float
    optimistic_init: float


@dataclass
class SimulationResult:
    method_key: str
    method_name: str
    avg_return_curve: np.ndarray
    ci95_curve: np.ndarray
    expected_total_reward: float
    regret: float
    allocation_dollars: np.ndarray


def tie_break_argmax(values: np.ndarray, rng: np.random.Generator) -> int:
    best = np.flatnonzero(values == values.max())
    return int(rng.choice(best))


def pull_reward(rng: np.random.Generator, arm_idx: int, means: np.ndarray) -> float:
    return 1.0 if rng.random() < means[arm_idx] else 0.0


def choose_arm(method: str, t: int, state: dict, rng: np.random.Generator, config: SimulationConfig) -> int:
    q = state["q"]
    n = state["n"]

    if method == "ab_test":
        half = config.explore_budget // 2
        if t < half:
            return 0
        if t < config.explore_budget:
            return 1
        if not state["winner_decided"]:
            mean_a = state["ab_sum"][0] / max(state["ab_n"][0], 1)
            mean_b = state["ab_sum"][1] / max(state["ab_n"][1], 1)
            state["winner"] = 0 if mean_a >= mean_b else 1
            state["winner_decided"] = True
        return state["winner"]

    if method == "optimistic":
        return tie_break_argmax(q, rng)

    if method == "epsilon_greedy":
        if t < 3:
            return t
        if rng.random() < config.epsilon:
            return int(rng.integers(0, 3))
        return tie_break_argmax(q, rng)

    if method == "softmax":
        tau = max(config.tau_min, config.tau0 * (config.tau_decay ** t))
        logits = q / max(tau, 1e-12)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return int(rng.choice(3, p=probs))

    if method == "ucb":
        for arm_idx in range(3):
            if n[arm_idx] == 0:
                return arm_idx
        bonus = config.ucb_c * np.sqrt(np.log(t + 1) / n)
        return tie_break_argmax(q + bonus, rng)

    if method == "thompson":
        samples = rng.beta(state["alpha"], state["beta"])
        return tie_break_argmax(samples, rng)

    raise ValueError(f"Unknown method: {method}")


def update_state(method: str, arm_idx: int, reward: float, state: dict, config: SimulationConfig) -> None:
    n = state["n"]
    q = state["q"]

    n[arm_idx] += 1
    q[arm_idx] += (reward - q[arm_idx]) / n[arm_idx]

    if method == "ab_test" and arm_idx in (0, 1) and state["ab_n"].sum() < config.explore_budget:
        state["ab_sum"][arm_idx] += reward
        state["ab_n"][arm_idx] += 1

    if method == "thompson":
        if reward > 0.5:
            state["alpha"][arm_idx] += 1
        else:
            state["beta"][arm_idx] += 1


def run_single_trajectory(method: str, rng: np.random.Generator, config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    means = np.array([config.mean_a, config.mean_b, config.mean_c], dtype=float)

    q_init = np.zeros(3, dtype=float)
    if method == "optimistic":
        q_init.fill(config.optimistic_init)

    state = {
        "q": q_init,
        "n": np.zeros(3, dtype=int),
        "alpha": np.ones(3, dtype=float),
        "beta": np.ones(3, dtype=float),
        "ab_sum": np.zeros(2, dtype=float),
        "ab_n": np.zeros(2, dtype=int),
        "winner_decided": False,
        "winner": 0,
    }

    rewards = np.zeros(config.total_budget, dtype=float)
    actions = np.zeros(config.total_budget, dtype=int)

    for t in range(config.total_budget):
        arm_idx = choose_arm(method, t, state, rng, config)
        reward = pull_reward(rng, arm_idx, means)
        actions[t] = arm_idx
        rewards[t] = reward
        update_state(method, arm_idx, reward, state, config)

    return rewards, actions


def simulate_method(method: str, config: SimulationConfig, seed: int) -> SimulationResult:
    rng = np.random.default_rng(seed)

    mean_cumsum = np.zeros(config.total_budget, dtype=float)
    m2_cumsum = np.zeros(config.total_budget, dtype=float)
    allocation_sum = np.zeros(3, dtype=float)

    for i in range(config.runs):
        rewards, actions = run_single_trajectory(method, rng, config)
        cumsum = np.cumsum(rewards)

        delta = cumsum - mean_cumsum
        mean_cumsum += delta / (i + 1)
        m2_cumsum += delta * (cumsum - mean_cumsum)

        allocation_sum += np.bincount(actions, minlength=3)

    std_cumsum = np.sqrt(m2_cumsum / (config.runs - 1)) if config.runs > 1 else np.zeros_like(mean_cumsum)

    spend = np.arange(1, config.total_budget + 1, dtype=float)
    avg_return_curve = mean_cumsum / spend
    se_curve = std_cumsum / np.sqrt(config.runs)
    ci95_curve = 1.96 * se_curve / spend

    expected_total_reward = float(mean_cumsum[-1])
    optimal_total_reward = config.total_budget * max(config.mean_a, config.mean_b, config.mean_c)
    regret = optimal_total_reward - expected_total_reward

    return SimulationResult(
        method_key=method,
        method_name=METHOD_LABELS[method],
        avg_return_curve=avg_return_curve,
        ci95_curve=ci95_curve,
        expected_total_reward=expected_total_reward,
        regret=regret,
        allocation_dollars=allocation_sum / config.runs,
    )


def run_all_methods(config: SimulationConfig, selected_methods: list[str]) -> list[SimulationResult]:
    results = []
    for i, method in enumerate(selected_methods):
        results.append(simulate_method(method, config, config.seed + i * 1000))
    results.sort(key=lambda x: x.expected_total_reward, reverse=True)
    return results


def plot_return_curve(results: list[SimulationResult], total_budget: int, show_ci_ab: bool) -> plt.Figure:
    spend = np.arange(1, total_budget + 1)
    fig, ax = plt.subplots(figsize=(11, 6))

    for r in results:
        ax.plot(spend, r.avg_return_curve, linewidth=1.8, label=r.method_name)
        if show_ci_ab and r.method_key == "ab_test":
            low = r.avg_return_curve - r.ci95_curve
            high = r.avg_return_curve + r.ci95_curve
            ax.fill_between(spend, low, high, alpha=0.2)

    ax.set_title("Average Return Rate vs Dollars Spent")
    ax.set_xlabel("Dollars Spent")
    ax.set_ylabel("Average Return Rate")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    return fig


def plot_bar_charts(results: list[SimulationResult], optimal_total_reward: float) -> plt.Figure:
    names = [r.method_name for r in results]
    rewards = [r.expected_total_reward for r in results]
    regrets = [r.regret for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, rewards)
    axes[0].axhline(optimal_total_reward, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_title("Expected Total Reward")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, regrets)
    axes[1].set_title("Regret (Optimal - Method)")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    return fig


def make_summary_df(results: list[SimulationResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "Method": r.method_name,
                "Expected Reward": round(r.expected_total_reward, 2),
                "Regret": round(r.regret, 2),
                "A($)": round(float(r.allocation_dollars[0]), 1),
                "B($)": round(float(r.allocation_dollars[1]), 1),
                "C($)": round(float(r.allocation_dollars[2]), 1),
            }
        )
    return pd.DataFrame(rows)


def app() -> None:
    st.set_page_config(page_title="6 Bandit Strategies", layout="wide")
    st.title("In-Class Activity: Comparing 6 Bandit Strategies")
    st.write("Use the sidebar to tune parameters, then run Monte Carlo simulation.")

    with st.sidebar:
        st.header("Simulation Settings")
        total_budget = st.number_input("Total budget", min_value=1000, max_value=100000, value=10000, step=500)
        default_explore = min(2000, int(total_budget) // 2)
        explore_budget = st.number_input(
            "A/B exploration budget",
            min_value=100,
            max_value=int(total_budget - 100),
            value=default_explore,
            step=100,
        )
        runs = st.slider("Monte Carlo runs", min_value=100, max_value=5000, value=600, step=100)
        seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=42, step=1)

        selected_labels = st.multiselect(
            "Methods to run",
            options=[METHOD_LABELS[m] for m in METHODS],
            default=[METHOD_LABELS[m] for m in METHODS],
        )

        st.header("True Means")
        mean_a = st.slider("Mean A", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
        mean_b = st.slider("Mean B", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
        mean_c = st.slider("Mean C", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

        st.header("Algorithm Hyperparameters")
        epsilon = st.slider("Epsilon (epsilon-greedy)", min_value=0.0, max_value=0.5, value=0.10, step=0.01)
        tau0 = st.slider("Softmax tau0", min_value=0.05, max_value=1.0, value=0.25, step=0.01)
        tau_min = st.slider("Softmax tau_min", min_value=0.01, max_value=0.2, value=0.03, step=0.01)
        tau_decay = st.slider("Softmax tau_decay", min_value=0.99, max_value=1.0, value=0.9995, step=0.0001)
        ucb_c = st.slider("UCB c", min_value=0.1, max_value=4.0, value=2.0, step=0.1)
        optimistic_init = st.slider("Optimistic initial value", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

        show_ci_ab = st.checkbox("Show 95% CI for A/B curve", value=True)

        run_button = st.button("Run Simulation", type="primary")

    config = SimulationConfig(
        mean_a=float(mean_a),
        mean_b=float(mean_b),
        mean_c=float(mean_c),
        total_budget=int(total_budget),
        explore_budget=int(explore_budget),
        runs=int(runs),
        seed=int(seed),
        epsilon=float(epsilon),
        tau0=float(tau0),
        tau_min=float(tau_min),
        tau_decay=float(tau_decay),
        ucb_c=float(ucb_c),
        optimistic_init=float(optimistic_init),
    )

    if run_button:
        if not selected_labels:
            st.error("Please select at least one method.")
            return

        label_to_method = {v: k for k, v in METHOD_LABELS.items()}
        selected_methods = [label_to_method[label] for label in selected_labels]

        estimated_steps = config.runs * config.total_budget * len(selected_methods)
        st.caption(f"Estimated simulation steps: {estimated_steps:,}")
        if estimated_steps > 40_000_000:
            st.warning("Large workload detected. Consider reducing runs or selected methods for faster response.")

        progress = st.progress(0)
        status = st.empty()

        with st.spinner("Running simulation..."):
            results = []
            for i, method in enumerate(selected_methods):
                status.text(f"Running {METHOD_LABELS[method]} ({i + 1}/{len(selected_methods)})")
                results.append(simulate_method(method, config, config.seed + i * 1000))
                progress.progress((i + 1) / len(selected_methods))

            results.sort(key=lambda x: x.expected_total_reward, reverse=True)

        status.text("Simulation complete")

        optimal_total_reward = config.total_budget * max(config.mean_a, config.mean_b, config.mean_c)
        st.subheader("Summary Table")
        st.dataframe(make_summary_df(results), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal Expected Reward", f"{optimal_total_reward:.2f}")
        col2.metric("Best Method", results[0].method_name)
        col3.metric("Best Expected Reward", f"{results[0].expected_total_reward:.2f}")

        st.subheader("Performance Plot")
        fig_curve = plot_return_curve(results, config.total_budget, show_ci_ab)
        st.pyplot(fig_curve)

        st.subheader("Reward / Regret Comparison")
        fig_bar = plot_bar_charts(results, optimal_total_reward)
        st.pyplot(fig_bar)

        st.subheader("Class Discussion Hints")
        st.markdown(
            "- A/B Testing is simple but static; it cannot adapt during exploration.\n"
            "- Epsilon-Greedy and Softmax are good baselines.\n"
            "- UCB and Thompson usually provide a better reward-regret balance."
        )
    else:
        st.info("Adjust settings in the sidebar, then click 'Run Simulation'.")


if __name__ == "__main__":
    app()