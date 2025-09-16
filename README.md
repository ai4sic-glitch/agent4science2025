Prerequisites
- conda environment manager
- Python 3.8+


Step 1. Setup conda environment
```
cd /path/to/your/project
conda env create -f environment.yml -n myenv
conda activate myenv
```
Step 2. Put your deepseek api key in utils/deepseek_api.py
```
DEFAULT_API_KEY = ""
```
Step 3. Change your dataset and task description at the end of agents/analysis_agent.py
```
csv_path = "./data/vineyard_multivariate_dataset.csv"
```
```
user_description = ("The goal is to predict redvine disease in 2024. The resulting model will be used to predict future redvine disease.")
```
Step 4. In your project folder, run:
```
python agents/analysis_agent.py
```
