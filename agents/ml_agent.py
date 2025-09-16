import pandas as pd
import subprocess
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from utils.deepseek_api import query_deepseek

MAX_ATTEMPTS = 5
GENCODE_DIR = "gen_code"
os.makedirs(GENCODE_DIR, exist_ok=True)
# OUTPUT_SCRIPT = "generated_analysis.py"

def inspect_dataset(csv_path):
    df = pd.read_csv(csv_path)
    columns = df.columns.tolist()
    # sample_rows = df.head(20).to_dict(orient='records')
    num_sample_rows = 20
    sampled_df = df.iloc[::50].head(num_sample_rows)
    sample_rows = sampled_df.to_dict(orient='records')
    return columns, sample_rows

def build_prompt(csv_file, columns, sample_rows, features, label, note, error_message=None):
    prompt = f"""
        You are an expert Python programmer and machine learning engineer. Your task is to write complete and correct Python code that analyzes tabular data for machine learning.

        Analyze dataset from {csv_file}.
        Suggested features: {features}.
        Target label: {label}.

        Dataset structure:
        Columns: {columns}
        Sample rows:
        {sample_rows}

        Analysis suggestion from analyst:
        "{note}"

        TASK:
        - Generate complete runnable Python code.
        - Handle numeric fields with units (e.g., "133ft" → 133.0) intelligently.
        - Preprocess features properly.
        - Train a model using auto-sklearn.
        - Set memory_limit for auto-sklearn to 4096 MB or less.
        - Print the actual features used and the evaluation report after training and testing.
        - Ensure the entire logic is inside a `main()` function and invoked with `if __name__ == "__main__": main()` for multiprocessing compatibility.
        
        Important: For any operation that involves spatial or distance-based features, or comparing all rows to each other, do NOT use double for-loops (O(N²) time complexity). Instead, use efficient vectorized, tree-based, or library-based methods (e.g., scikit-learn BallTree, KDTree, or numpy vectorization). Always avoid solutions that will be slow for large datasets.
        Important: Only output plain Python code. Do not include any explanatory text, markdown formatting, or usage instructions. The output must be directly executable as a `.py` script.
    """
    if error_message:
        prompt += f"\nNote: The previous attempt failed with this error:\n{error_message}\nPlease fix the issue and regenerate complete working code."
    return prompt

def clean_code_output(raw_text):
    lines = raw_text.strip().splitlines()
    lines = [line for line in lines if not line.strip().startswith("```")]
    return "\n".join(lines)

def save_script(code):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    filename = os.path.join(GENCODE_DIR, f"generated_analysis_{timestamp}.py")
    with open(filename, "w") as f:
        f.write(code)
    print(f"Saved generated code to {filename}")
    return filename

def run_script(filename):
    try:
        result = subprocess.run(
            ["python", filename],
            capture_output=True, # Stream the output (comment this line) for testing purposes
            text=True,
            timeout=1200  # 20-minute hard timeout
        )
        return result
    except subprocess.TimeoutExpired:
        print(f"Script {filename} timed out after 20 minutes.")
        return subprocess.CompletedProcess(args=["python", filename], returncode=1, stdout="", stderr="Timeout")

def ml_agent(csv_file, features, label, note):
    columns, sample_rows = inspect_dataset(csv_file)
    attempt = 0
    error_message = "Errors encountered in previous attempts:\n"

    while attempt < MAX_ATTEMPTS:
        print(f"ML Agent Coding Attempt {attempt+1}...")
        prompt = build_prompt(csv_file, columns, sample_rows, features, label, note, error_message)
        messages = [
            {"role": "system", "content": "You are an expert Python programmer and machine learning engineer. Your task is to generate runnable Python code that can preprocess and analyze tabular datasets."},
            {"role": "user", "content": prompt}
        ]
        try:
            generated_code = query_deepseek(messages)
            # generated_code = query_deepseek(messages, model="deepseek-reasoner") # deepseek-reasoner
            # print("\nResponse from DeepSeek:\n", generated_code)
        except Exception as e:
            print(f"DeepSeek API call failed: {e}")
            return None

        generated_code = clean_code_output(generated_code)
        filename = save_script(generated_code)
        print(f"Running generated script...")
        result = run_script(filename)
        if result.returncode == 0:
            print("Model trained successfully.")
            clean_stdout = "\n".join(
                line for line in result.stdout.splitlines()
                if "[WARNING]" not in line and "RunKey" not in line
            )  # if "[WARNING]" not in line or "EnsembleBuilder" not in line
            return clean_stdout
        else:
            print(f"Script failed with error:\n{result.stderr}")
            error_message += result.stderr
            error_message += "\n"
            attempt += 1
            # print(error_message)

    print("All attempts failed after maximum retries.")
    return 'All attempts failed after maximum retries.'

if __name__ == "__main__":
    csv_file = "./data/vineyard_multivariate_dataset_with_coords_djmod.csv"
    features = ['redvine_count_2021', 'redvine_count_2022', 'redvine_count_2023', 
                'EVI_2022-06-01', 'EVI_2022-07-06', 'EVI_2022-08-04', 
                'CanopyArea_2022-06-01', 'CanopyArea_2022-07-06', 'CanopyArea_2022-08-04',
                'longitude', 'latitude']
    label = 'redvine_count_2024'
    note = "Handle missing values (many NaN values in 2023-2025 columns), consider temporal aggregation of EVI/Canopy metrics, and standardize numeric features. Geographic coordinates may need transformation for spatial modeling."
    ml_agent(csv_file, features, label, note)