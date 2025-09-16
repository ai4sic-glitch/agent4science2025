import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.deepseek_api import query_deepseek
from agents.ml_agent import ml_agent
from agents.domain_agent import evaluate_experiment_biologically
from datetime import datetime

### Log Output 2>&1 ###
timestamp = datetime.now().strftime("%m%d_%H%M%S")
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log_{timestamp}.txt")

log_f = open(log_file, "w")
sys.stdout = log_f
sys.stderr = log_f
#######################

def summarize_experiment(history, user_description):
    history_text = "\n\n".join(history)
    prompt = (
        "Summarize the outcome of a machine learning experiment.\n\n"
        f"User description: {user_description}\n\n"
        f"Experiment history:\n{history_text}\n\n"
        "Provide a concise, insightful summary including:\n"
        "- Overall performance interpretation\n"
        "- Class-wise strengths/weaknesses\n"
        "- Key limitations of this experiment\n"
        "- Suggestions for future improvement\n\n"
        "Respond in plain text."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert machine learning analyst. Your job is to summarize results, interpret performance, and provide suggestions."
            )
        },
        {"role": "user", "content": prompt}
    ]

    try:
        summary = query_deepseek(messages)
        print(f"\n=== Experiment history ===\n{history_text}")
        print("\n=== Final summary report ===")
        print(summary)
    except Exception as e:
        print(f"Error during final summary query: {e}")

def analysis_loop(csv_path, user_description, deepseek_model, max_iterations):
    df = pd.read_csv(csv_path)
    columns = df.columns.tolist()
    
    num_sample_rows = 20
    sampled_df = df.iloc[::50].head(num_sample_rows)
    sample_rows = sampled_df.to_dict(orient='records')

    iteration = 0
    satisfied = False
    bio_evaluation = None
    last_report = None
    label = None
    features = None
    note = None
    history = []
    leaderboard = []

    while iteration < max_iterations and not satisfied:
        print(f"\n=== Analysis iteration {iteration+1} ===")

        if iteration == 0:
            # Initial reasoning about dataset structure
            prompt = (
                f"This is iteration {iteration+1} out of {max_iterations}.\n\n"
                "I have a tabular dataset for machine learning analysis.\n\n"
                f"User description of desired analysis:\n{user_description}\n\n"
                f"Columns: {columns}\n\n"
                f"Sample rows: {sample_rows}\n\n"
                "Your tasks:\n"
                "1. Identify the most suitable column to predict (label/target).\n"
                "2. Recommend the most important numeric features for prediction.\n"
                "3. Suggest a short preprocessing note, mentioning any issues you notice such as mixed units, missing values, or formatting problems.\n"
                "4. Decide whether this task is better framed as classification or regression based on the target column, and include that reasoning in the Note.\n\n"
                "Respond exactly in the following plain-text format (do not use Markdown, bold, italics, or any other formatting):\n"
                "Label: <column name>\n"
                "Features: <comma-separated column names>\n"
                "Note: <brief preprocessing suggestion>\n"
                "The reason for this modeling suggestion: <brief explanation of why these features and notes are important>"
            )
        else:
            # Review result and suggest refinement
            # Compile full history text
            history_text = "\n\n".join(history)
            prompt = (
                f"This is iteration {iteration+1} out of {max_iterations}.\n\n"
                f"User description of desired analysis:\n{user_description}\n\n"
                f"Columns: {columns}\n\n"
                f"Sample rows: {sample_rows}\n\n"
                f"Experiment history so far:\n{history_text}\n\n"
                "You are refining an ongoing machine learning modeling experiment. Think through the following steps carefully:\n"
                "1. First, decide whether the latest model performance is already sufficient for the intended use. If it is, stop here and finalize the result.\n"
                "2. If not, examine the full experiment history carefully — including previous feature selections, preprocessing strategies, and modeling results.\n"
                "3. Identify which decisions or techniques appear promising (e.g., feature combinations, preprocessing steps, target framing) and should be kept or explored further. Avoid restarting from scratch or reusing previous combinations that have already performed poorly.\n"
                "4. Based on this, suggest a revised set of features and a modeling strategy. If you propose new features (e.g., derived features, transformed values), describe them clearly in the Note. Do not invent new features that cannot be computed from existing columns in the dataset.\n"
                "5. Early in the experiment, explore both classification and regression modeling strategies if either could be appropriate. Use evidence from past iterations to decide which framing better aligns with the task and model performance. State your reasoning and final choice in the Note.\n"
                "6. Consider time budget management. You have approximately 15 minutes in total for one task. In the Note, suggest a time limit that balances search space and efficiency. General recommendation is between 120 to 600 seconds. Recommend turning off ensemble building unless this round is very promising and you are aiming for a final, high-performance model.\n\n"
                "Respond exactly in the following plain-text format (do not use Markdown, bold, italics, or any other formatting):\n"
                "Decision: <'satisfied' or 'continue'>\n"
                "Features: <comma-separated columns if 'continue'; 'same' if no change>\n"
                "Note: <brief suggestion if 'continue'; empty if 'satisfied'>\n"
                "The reason for this modeling suggestion: <brief explanation of why these features and notes are important>"
            )

            # print(f"\nPrompt for analysis agent:\n{prompt}")
            # print(f"Experiment history so far:\n{history_text}\n")
            # print(f"Latest evaluation report from a plant pathology scientist:\n{bio_evaluation}\n")

        messages = [
            {"role": "system", "content": (
                    "You are an autonomous machine learning analyst."
                    "You are an expert agricultural and plant biology data analyst."
                    "You help users analyze any dataset for machine learning modelling by suggesting appropriate label, features, and preprocessing tips."
                )
            },
            {"role": "user", "content": prompt}
        ]

        try:
            response = query_deepseek(messages, model=deepseek_model) #TODO response = query_llm(messages, model=openai/deepseek/...)
            print(f"\nResponse from DeepSeek:\n{response}")
        except Exception as e:
            print(f"Error during DeepSeek query: {e}")
            break

        if iteration == 0:
            # Parse initial suggestion
            label_line = next(line for line in response.splitlines() if line.startswith("Label:"))
            features_line = next(line for line in response.splitlines() if line.startswith("Features:"))
            note_line = next(line for line in response.splitlines() if line.startswith("Note:"))
            reason_line = next(line for line in response.splitlines() if line.startswith("The reason for this modeling suggestion:"))

            label = label_line.split("Label:")[1].strip()
            features = [f.strip() for f in features_line.split("Features:")[1].split(",")]
            note = note_line.split("Note:")[1].strip()
            reason = reason_line.split("The reason for this modeling suggestion:")[1].strip()

        else:
            decision_line = next(line for line in response.splitlines() if line.startswith("Decision:"))
            decision = decision_line.split("Decision:")[1].strip().lower()

            if decision == "satisfied":
                print("Agent is satisfied with current model performance. Stopping loop.")
                satisfied = True
                reason_line = next(line for line in response.splitlines() if line.startswith("The reason for this modeling suggestion:"))
                reason = reason_line.split("The reason for this modeling suggestion:")[1].strip()
                iteration_record = (
                    f"Iteration {iteration+1}:\n"
                    f"Agent is satisfied with current model performance. Stopping loop.\n"
                    f"Reasons for satisfaction: {reason}\n"
                )
                history.append(iteration_record)
                break

            features_line = next(line for line in response.splitlines() if line.startswith("Features:"))
            note_line = next(line for line in response.splitlines() if line.startswith("Note:"))
            reason_line = next(line for line in response.splitlines() if line.startswith("The reason for this modeling suggestion:"))

            new_features = features_line.split("Features:")[1].strip()
            if new_features != "same":
                features = [f.strip() for f in new_features.split(",")]
            # If "same", retain current `features`.

            note = note_line.split("Note:")[1].strip()
            reason = reason_line.split("The reason for this modeling suggestion:")[1].strip()

        # Run ML agent for current iteration
        # print(f"\nRunning ml_agent with features: {features}, label: {label}, note: {note}")
        report = ml_agent(csv_path, features, label, note)
        print("\nGenerated report from ML agent:\n", report)
        last_report = report

        bio_evaluation =  evaluate_experiment_biologically(user_description=user_description, columns=columns, sample_rows=sample_rows, features=features, ml_report=last_report)
        print(f"\nBiological evaluation:\n{bio_evaluation}")

        # Record iteration in history
        iteration_record = (
            f"Iteration {iteration+1}:\n"
            f"Label selected: {label}\n"
            f"Features selected: {', '.join(features) if features else 'None'}\n"
            f"Preprocessing note: {note}\n"
            f"Reasons for modeling suggestion: {reason}\n"
            f"Mechine Learning modeling result:\n{last_report}\n"
            f"Biological evaluation:\n{bio_evaluation}"
        )
        leaderboard_record = (
            f"Iteration {iteration+1} |"
            f"Features: {', '.join(features) if features else 'None'}\n"
            f"Report: {last_report}\n"
        )
        # print(f"\nIteration record:\n{iteration_record}")
        history.append(iteration_record)
        leaderboard.append(leaderboard_record)
        
        iteration += 1

    if not satisfied:
        print("\nMax iterations reached. Stopping without explicit satisfaction.")

    # Final summary of the experiment
    if last_report:
        summarize_experiment(history, user_description)

def main():
    csv_path = "./data/08_multiscale_features_simple_clean.csv"
    user_description = ("The goal is to predict redvine disease in 2024 using data before Aug 2024. The resulting model will be used to predict future redvine disease. The presence column in the dataset is an average presence/absence of grape vine at grid points, no matter infected or not.")
    
    deepseek_model="deepseek-chat"
    # deepseek_model="deepseek-reasoner"

    max_iterations=20

    print(f"Starting analysis loop for CSV: {csv_path}")
    print(f"User description: {user_description}")
    print(f"Using DeepSeek model: {deepseek_model}")
    print(f"Max iterations allowed: {max_iterations}")

    analysis_loop(csv_path, user_description, deepseek_model, max_iterations) 

if __name__ == "__main__":
    main()
