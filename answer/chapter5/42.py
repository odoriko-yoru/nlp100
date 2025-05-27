"""多肢選択問題の正解率."""

import csv
import os
from pathlib import Path
from typing import Union

import anthropic
from tqdm import tqdm

path = os.environ.get("DATA_DIR", "")
path = Path(path).parent

csv_file = "JMMLU/JMMLU/college_biology.csv"

filepath = path / csv_file


def load_jmmlu_csv(filepath: Union[str, Path]) -> list[dict]:
    """Load the JMMLU QA Dataset.

    Parameters
    ----------
    filepath : Union[str, Path]
        file path to csv file.

    Returns
    -------
    list[dict]
    """
    qa = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 6:
                r = {"question": row[0], "choices": row[1:5], "answer": row[5]}
                qa.append(r)
        return qa


def create_prompt_from_dict(problem: dict[str, str]) -> str:
    """Create a prompt from inputted problem from JMMLU.

    Parameters
    ----------
    problem : dict[str, str]
        Multiple choice problem.

    Returns
    -------
    str
        Prompt
    """
    prompt = f"""
    以下に問題文とA, B, C, Dからなる選択肢を与えます。
    問題文に対して正しい選択肢をA-Dから選択してください。

    解答はA, B, C, Dの記号のみを返してください。

    問題
    {problem["question"]}


    選択肢
    A. {problem["choices"][0]}
    B. {problem["choices"][1]}
    C. {problem["choices"][2]}
    D. {problem["choices"][3]}
    """

    return prompt


client = anthropic.Anthropic()

# Hyperparameter
# model = "claude-3-7-sonnet-latest"
model = "claude-3-5-haiku-latest"
max_tokens = 1024
temperature = 0

dataset = load_jmmlu_csv(filepath)

correct = 0

for problem in tqdm(dataset):
    prompt = create_prompt_from_dict(problem)
    gt = problem["answer"]
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    ans = message.content[0].text
    correct += gt == ans

print(f"正解率 : {correct / len(dataset) * 100:.4f}%")
print(f"正解率/問題数 : {correct} / {len(dataset)}")
