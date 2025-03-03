import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        ## Structure Format Reward Function
        
        This example demonstrates a reward function that evaluates whether 
        completions follow a specific structure format. Either a `<think>...</think><answer>...</answer>`
        or a `<code>...</code><explanation>...</explanation>` format.
        
        Use the buttons to select which structure format to reward.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    format_buttons = mo.ui.radio(
        options=["think-answer", "code-explanation"],
        value="think-answer",
        label="Format to reward",
    )
    format_buttons
    return (format_buttons,)


@app.cell(hide_code=True)
def _(mo, format_buttons):
    import plotly.express as px
    import re

    # Sample completions with different formats
    completions = [
        # Think-answer format examples
        "<think>Let me solve this step by step</think><answer>42</answer>",
        "The answer is 15 without any special format",
        "<code>print('Hello world')</code><explanation>This prints a greeting</explanation>",
        # Code-explanation format examples
        "<code>def add(a, b): return a + b</code><explanation>A function to add numbers</explanation>",
        "<code>for i in range(10): print(i)</code>",
        "<think>I should use a loop</think><code>while True: pass</code>",
    ]

    # Create shortened versions for display
    short_completions = [c[:30] + "..." if len(c) > 30 else c for c in completions]

    def format_reward(completions, format_type="think-answer", **kwargs):
        """
        Reward completions that follow the desired format structure

        Args:
            completions: list of completions to evaluate
            format_type: which format structure to reward

        Returns:
            list of rewards and details
        """
        # Define patterns for different formats
        patterns = {
            "think-answer": r"<think>.*?</think>\s*<answer>.*?</answer>",
            "code-explanation": r"<code>.*?</code>\s*<explanation>.*?</explanation>",
        }

        # Select the pattern based on format_type
        pattern = patterns.get(format_type, patterns["think-answer"])

        rewards = []
        details = []
        categories = []

        for completion in completions:
            match = re.search(pattern, completion, re.DOTALL)
            if match:
                # Full match for the exact format
                rewards.append(1.0)
                details.append(f"Correct {format_type} format")
                categories.append("Exact Format Match")
            elif f"<{format_type.split('-')[0]}>" in completion:
                # Partial match - has the opening tag of the format
                rewards.append(0.5)
                details.append(f"Has {format_type.split('-')[0]} tag but incomplete")
                categories.append("Partial Format Match")
            elif any(f"<{tag}>" in completion for tag in format_type.split("-")):
                # Contains at least one of the required tags
                rewards.append(0.2)
                details.append("Has some required tags but wrong format")
                categories.append("Some Tags Present")
            else:
                # No match at all
                rewards.append(0.0)
                details.append("Incorrect format")
                categories.append("No Format Match")

        return rewards, details, categories

    # Calculate rewards
    rewards, details, categories = format_reward(
        completions=completions, format_type=format_buttons.value
    )

    # Display the results
    results = []
    for completion, reward, detail, category in zip(
        short_completions, rewards, details, categories
    ):
        results.append(
            {
                "Completion": completion,
                "Reward": reward,
                "Detail": detail,
                "Category": category,
            }
        )

    # Create a table view
    mo.md(f"### Results for {format_buttons.value} format")
    mo.ui.table(results)

    # Create a bar chart comparing rewards by completion
    fig = px.bar(
        results,
        x="Completion",
        y="Reward",
        color="Category",
        title=f"Format Rewards by Completion ({format_buttons.value})",
        hover_data=["Detail"],
    )
    mo.ui.plotly(fig)


if __name__ == "__main__":
    app.run()
