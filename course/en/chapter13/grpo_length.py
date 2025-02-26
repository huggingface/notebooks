import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        "## Length based reward\nAdjust the slider to see how the reward changes for different completion lengths."
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    slider = mo.ui.slider(start=5, stop=50, step=5, label="Ideal Length (characters)")
    slider
    return (slider,)


@app.cell(hide_code=True)
def _(mo, slider):
    import plotly.express as px

    # Toy dataset with 5 samples of different lengths
    completions = [
        "Short",  # 5 chars
        "Medium length text",  # 18 chars
        "This is about twenty chars",  # 25 chars
        "This is a slightly longer completion",  # 36 chars
        "This is a much longer completion with more words",  # 45 chars
    ]

    maximum_length = max(len(completion) for completion in completions)
    minimum_length = min(len(completion) for completion in completions)

    def length_reward(completions, ideal_length):
        """
        Calculate rewards based on the length of completions.

        Args:
            completions: List of text completions
            ideal_length: Target length in characters

        Returns:
            List of reward scores for each completion
        """
        rewards = []

        for completion in completions:
            length = len(completion)
            # Simple reward function: negative absolute difference
            reward = maximum_length - abs(length - ideal_length)
            reward = max(0, reward)
            reward = min(1, reward / (maximum_length - minimum_length))
            rewards.append(reward)

        return rewards

    # Calculate rewards for the examples
    rewards = length_reward(completions=completions, ideal_length=slider.value)

    # Display the examples and their rewards
    results = []
    for completion, reward in zip(completions, rewards):
        results.append(
            {"Completion": completion, "Length": len(completion), "Reward": reward}
        )

    fig = px.bar(results, x="Completion", y="Reward", color="Length")
    mo.ui.plotly(fig)


if __name__ == "__main__":
    app.run()
