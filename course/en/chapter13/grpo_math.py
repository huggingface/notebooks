import marimo

__generated_with = "0.10.6"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        ## Math Problem Reward Function
        
        This example demonstrates a reward function for math problems 
        with verifiable answers.
        
        The slider controls the tolerance for approximate answers.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    tolerance_slider = mo.ui.slider(
        start=0, stop=25, step=5, value=0, label="Tolerance"
    )
    tolerance_slider
    return (tolerance_slider,)


@app.cell(hide_code=True)
def _(mo, tolerance_slider):
    import plotly.express as px

    # Sample math problems and their correct answers
    problems = [
        "What is 5 + 7?",
        "Calculate 12 * 6",
        "What is 100 / 4?",
        "Solve for x: 3x = 15",
        "What is the square root of 81?",
    ]

    # Correct answers
    correct_answers = [12, 72, 25, 5, 9]

    # Model completions (simulated)
    model_completions = [
        12,  # Correct
        92,  # Wrong
        15,  # Wrong
        0,  # Wrong
        9,  # Correct
    ]

    def extract_final_answer(completion):
        """
        In a real scenario, this would parse the completion to extract the answer.
        For this example, we're using direct integer completions.
        """
        return completion

    def problem_reward(completions, answers, tolerance=0):
        """
        Reward function for math problems with verifiable answers

        Args:
            completions: list of completions to evaluate
            answers: list of correct answers to the problems
            tolerance: allowed difference for correct answers

        Returns:
            list of rewards for each completion
        """
        rewards = []

        for completion, correct_answer in zip(completions, answers):
            try:
                # Extract the answer from the completion
                answer = extract_final_answer(completion)

                # Calculate how close the answer is
                difference = abs(answer - correct_answer)

                # Binary reward with tolerance
                if difference <= tolerance:
                    reward = 1.0
                else:
                    # Partial credit based on how close the answer is
                    # Scale with problem size
                    max_diff = max(correct_answer * 0.5, 10)
                    reward = max(0, 1 - (difference / max_diff))

                rewards.append(reward)
            except Exception:
                # If we can't parse an answer, give a low reward
                rewards.append(0.0)

        return rewards

    # Calculate rewards
    rewards = problem_reward(
        completions=model_completions,
        answers=correct_answers,
        tolerance=tolerance_slider.value,
    )

    # Display the results
    results = []
    for problem, correct, completion, reward in zip(
        problems, correct_answers, model_completions, rewards
    ):
        results.append(
            {
                "Problem": problem,
                "Correct Answer": correct,
                "Model Answer": completion,
                "Difference": abs(correct - completion),
                "Reward": reward,
            }
        )

    # Create a table view
    mo.md("### Results")
    mo.ui.table(results)

    # Create a bar chart
    fig = px.bar(
        results,
        x="Problem",
        y="Reward",
        color="Difference",
        hover_data=["Correct Answer", "Model Answer"],
        title="Rewards by Problem",
    )
    mo.ui.plotly(fig)


if __name__ == "__main__":
    app.run()
