import os
import sys
import json
import fitz  # PyMuPDF
import re
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.deepseek_api import query_deepseek

# Constants
PAPER_DIR = Path("papers")
CACHE_PATH = Path("cache/biology_knowledge.json")


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


def summarize_paper(text):
    messages = [
        {"role": "system", "content": (
            "You are a plant pathology research assistant. "
            "Your task is to read scientific papers and extract structured biological knowledge useful for modeling grapevine disease prediction. "
            "Please extract key variables, biological relationships, and other relevant scientific insights. "
            "Respond in the following JSON format:\n\n"
            "{\n"
            "title: <title of the paper>,\n"
            "authors: <list of author names>,\n"
            "summary: <short summary of the main findings>,\n"
            "key_variables: [<list of biological variables relevant to modeling>],\n"
            "relationships: [<causal or correlative relationships among variables>]\n"
            "}\n\n"
            "Only output valid JSON, no explanations or extra text."
        )},
        {"role": "user", "content": f"Extract knowledge from this paper:\n{text[:6000]}"}
    ]
    response = query_deepseek(messages)

    match = re.search(r"```json(.*?)```", response, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        cleaned = response.strip()

    return json.loads(cleaned)


def update_knowledge_base():
    if not PAPER_DIR.exists():
        print("No paper folder found.")
        return

    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r") as f:
            existing = json.load(f)
            known_titles = {entry.get("title", "") for entry in existing}
    else:
        existing = []
        known_titles = set()

    updated = list(existing)
    for pdf in PAPER_DIR.glob("*.pdf"):
        try:
            text = extract_text_from_pdf(pdf)
            entry = summarize_paper(text)
            if entry.get("title") not in known_titles:
                entry["id"] = entry.get("title", pdf.stem)
                updated.append(entry)
                print(f"Added: {entry['title']}")
        except Exception as e:
            print(f"Error processing {pdf.name}: {e}")

    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(updated, f, indent=2)

    print("Knowledge base updated.")


def evaluate_experiment_biologically(user_description, columns, sample_rows, features, ml_report, max_lit=-1):
    if not CACHE_PATH.exists():
        return "Biology knowledge unavailable."

    with open(CACHE_PATH, "r") as f:
        knowledge = json.load(f)

    context = "\n\n".join(
        f"Title: {entry['title']}\n"
        f"Summary: {entry['summary']}\n"
        f"Key Variables: {', '.join(entry['key_variables'])}\n"
        f"Relationships: {', '.join(entry['relationships'])}"
        for entry in knowledge[:max_lit]
    )
    # print(f"Full context of plant path agent: {context}")

    prompt = (
        "You are a biology-aware evaluator of ML experiments.\n\n"
        f"Biological knowledge:\n{context}\n\n"
        f"User task:\n{user_description}\n\n"
        f"The dataset columns you have:\n{columns}\n\n"
        f"Sample rows:\n{sample_rows}\n\n"
        f"Selected features in this round for ML modeling:\n{features}\n\n"
        f"ML evaluation report in this round:\n{ml_report}\n\n"
        "Evaluate whether this result is biologically meaningful, and whether the dataset has been sufficiently utilized.\n"
        "Favor using fewer features with higher interpretability.\n"
        "If it is not, briefly explain why and suggest concise, biologically relevant improvements.\n"
        "You may consider domain-specific bias, when suggesting improvements for preprocessing.\n"
        "Limit your answer to under 200 words and do not end with a question."
    )

    messages = [
        {"role": "system", "content": "You are a plant pathologist."},
        {"role": "user", "content": prompt}
    ]

    return query_deepseek(messages)


if __name__ == "__main__":
    update_knowledge_base()

    # evaluation = evaluate_experiment_biologically(
    #     user_description="Evaluate the latest grapevine disease prediction model.",
    #     columns=["longitude", "latitude", "Altitude"],
    #     sample_rows=[["-123.456", "45.678", "100ft"]],
    #     features=["longitude", "latitude"],
    #     ml_report="Model accuracy: 85%, Precision: 80%, Recall: 75%",
    # )
    # print("Biological Evaluation:", evaluation)

