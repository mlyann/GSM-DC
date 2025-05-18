import os
import re
import json
import torch
import threading
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_gen.pretrain.id_gen import IdGen
from tools.tools import tokenizer, fix_seed
from tools.irr_tools_test import true_correct
from data_gen.prototype.id_gen import IdGen_PT
from typing import List
from prm_tree import tree_search
import sys
import re
from format import format_prompt

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "MODEL_PATH"

print(f"using {MODEL_PATH} to generate responses")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
model_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model_tokenizer.pad_token = model_tokenizer.eos_token  # Ensure pad token is set

def generate_response(problem):
    input_text = format_prompt(True, problem)
    inputs = model_tokenizer(
        input_text,
        return_tensors="pt",
    )
    input_len = inputs.input_ids.shape[1]
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024)

    generated_text = model_tokenizer.decode(outputs[0][input_len - 1:], skip_special_tokens=True)
    return generated_text.strip()


id_gen = IdGen_PT(
        style="light",
        op_style="light",
        max_op=15,
        perm_level=5,
        detail_level=0
    )

NUM_SAMPLES = 5000
data = []

for i in tqdm(range(NUM_SAMPLES), desc=f"Generating {NUM_SAMPLES} base samples"):
    id_gen.gen_prob([i for i in range(5)], p_format="pq")
    problem_text = tokenizer.decode(id_gen.prob_token)
    predicted_solution = generate_response(problem_text)
    irr_correct, correct, my_print, _ = true_correct(predicted_solution, id_gen.problem)

    steps = []
    start = 0
    end = 0
    while end < len(predicted_solution):
        if predicted_solution[end] == ";" or predicted_solution[end] == ".":
            steps.append(predicted_solution[start: end + 1].strip())
            end += 1
            start = end
        else:
            end += 1

    incorrect = torch.zeros((len(steps),))

    for i in range(len(my_print.txt)):
        text = my_print.txt[i][0]
        if "Mistake happened on param" in text:
            bad_param = text[26:].strip()
            
            for j in range(len(steps)):
                if bad_param in steps[j]:
                    labels = [0] * j + [1] * (len(steps) - j)
                    incorrect += torch.tensor(labels)
                    break
        if "existing_but_not_required_params:" in text:
            bad_params = text[33:].split(";")
            bad_params = [param.strip() for param in bad_params]
            found_bad_param = False

            for j in range(len(steps)):
                for bad_param in bad_params:
                    if bad_param != "" and bad_param in steps[j]:
                        labels = [0] * j + [1] * (len(steps) - j)
                        incorrect += torch.tensor(labels)
                        found_bad_param = True
                        break
                if found_bad_param:
                    break
        if "answered param" in text:
            incorrect += torch.tensor([0]*(len(steps)-2) + [1]*2)
        if "not in set()" in text:
            bad_sentence = text[:len(text) - 12].strip()
            
            for j in range(len(steps)):
                if bad_sentence in steps[j]:
                    labels = [0] * j + [1] * (len(steps) - j)
                    incorrect += torch.tensor(labels)
                    break
        if "Undefined symbol:" in text:
            bad_symbol = text[18:].strip()
            wrong_sentence = False

            for j in range(len(steps)):
                step = steps[j]
                for k in range(len(step) - len(bad_symbol) + 1):
                    if step[k: k + len(bad_symbol)] == bad_symbol and (k - 1 < 0 or step[k - 1] == " ") and (k + 1 >= len(step) or step[k + 1] == " "):
                        wrong_sentence = True
                        break
                
                if wrong_sentence:
                    labels = [0] * j + [1] * (len(steps) - j)
                    incorrect += torch.tensor(labels)
                    break
        if "Duplicated symbol:" in text:
            bad_symbol = text[18:].strip()
            seen_symbol = False

            for j in range(len(steps)):
                step = steps[j]

                if step[len(step) - 1] == ";":
                    if step[len(step) - len(bad_symbol) - 1 : len(step) - 1].strip() == bad_symbol:
                        if seen_symbol == True:
                            labels = [0] * j + [1] * (len(steps) - j)
                            incorrect += torch.tensor(labels)
                            break
                        else:
                            seen_symbol = True
        #first occurence?
        if "Illegal def part" in text:
            bad_part = text[17:].strip()
            
            for j in range(len(steps)):
                if bad_part in steps[j]:
                    labels = [0] * j + [1] * (len(steps) - j)
                    incorrect += torch.tensor(labels)
                    break
        if "has already been defined" in text:
            bad_part = text[:len(text) - 25].strip()

            for j in range(len(steps)):
                if bad_part in steps[j]:
                    labels = [0] * j + [1] * (len(steps) - j)
                    incorrect += torch.tensor(labels)
                    break
        if "none appear lst" in text or "Not a single valid sentence is found" in text:
            incorrect += torch.tensor([1]*len(steps))
        if "Wrong solution answer" in text:
            incorrect += torch.tensor([0]*(len(steps) - 1) + [1])


    labels = incorrect.tolist()
    labels = ["+" if i == 0 else "-" for i in labels]

    data_point = {
        "input": f"<|start_header_id|>user<|end_header_id|>\n\n"+problem_text+f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "value": steps,
        "label": labels
        }
    data.append(data_point)

id_gen = IdGen_PT(
        style="light",
        op_style="light",
        op=15,
        perm_level=5,
        detail_level=0
    )

FIXED_NUM_SAMPLES = 1000

for i in tqdm(range(FIXED_NUM_SAMPLES), desc=f"Generating {FIXED_NUM_SAMPLES} fixed samples"):
    id_gen.gen_prob([i for i in range(5)], p_format="pq")
    problem_text = tokenizer.decode(id_gen.prob_token)
    predicted_solution = generate_response(problem_text)
    irr_correct, correct, my_print, _ = true_correct(predicted_solution, id_gen.problem)

    steps = []
    start = 0
    end = 0
    while end < len(predicted_solution):
        if predicted_solution[end] == ";" or predicted_solution[end] == ".":
            steps.append(predicted_solution[start: end + 1].strip())
            end += 1
            start = end
        else:
            end += 1

    incorrect = torch.zeros((len(steps),))

    for i in range(len(my_print.txt)):
        text = my_print.txt[i][0]
        if "Mistake happened on param" in text:
            bad_param = text[26:].strip()
            
            for j in range(len(steps)):
                if bad_param in steps[j]:
                    labels = [0] * j + [1] * (len(steps) - j)
                    incorrect += torch.tensor(labels)
                    break
        if "existing_but_not_required_params:" in text:
            bad_params = text[33:].split(";")
            bad_params = [param.strip() for param in bad_params]
            found_bad_param = False

            for j in range(len(steps)):
                for bad_param in bad_params:
                    if bad_param != "" and bad_param in steps[j]:
                        labels = [0] * j + [1] * (len(steps) - j)
                        incorrect += torch.tensor(labels)
                        found_bad_param = True
                        break
                if found_bad_param:
                    break
        if "answered param" in text:
            incorrect += torch.tensor([0]*(len(steps)-2) + [1]*2)
        if "not in set()" in text:
            bad_sentence = text[:len(text) - 12].strip()
            
            for j in range(len(steps)):
                if bad_sentence in steps[j]:
                    labels = [0] * j + [1] * (len(steps) - j)
                    incorrect += torch.tensor(labels)
                    break
        if "Undefined symbol:" in text:
            bad_symbol = text[18:].strip()
            wrong_sentence = False

            for j in range(len(steps)):
                step = steps[j]
                for k in range(len(step) - len(bad_symbol) + 1):
                    if step[k: k + len(bad_symbol)] == bad_symbol and (k - 1 < 0 or step[k - 1] == " ") and (k + 1 >= len(step) or step[k + 1] == " "):
                        wrong_sentence = True
                        break
                
                if wrong_sentence:
                    labels = [0] * j + [1] * (len(steps) - j)
                    incorrect += torch.tensor(labels)
                    break
        if "Duplicated symbol:" in text:
            bad_symbol = text[18:].strip()
            seen_symbol = False

            for j in range(len(steps)):
                step = steps[j]

                if step[len(step) - 1] == ";":
                    if step[len(step) - len(bad_symbol) - 1 : len(step) - 1].strip() == bad_symbol:
                        if seen_symbol == True:
                            labels = [0] * j + [1] * (len(steps) - j)
                            incorrect += torch.tensor(labels)
                            break
                        else:
                            seen_symbol = True
        if "Illegal def part" in text:
            bad_part = text[17:].strip()
            
            for j in range(len(steps)):
                if bad_part in steps[j]:
                    labels = [0] * j + [1] * (len(steps) - j)
                    incorrect += torch.tensor(labels)
                    break
        if "has already been defined" in text:
            bad_part = text[:len(text) - 25].strip()

            for j in range(len(steps)):
                if bad_part in steps[j]:
                    labels = [0] * j + [1] * (len(steps) - j)
                    incorrect += torch.tensor(labels)
                    break
        if "none appear lst" in text or "Not a single valid sentence is found" in text:
            incorrect += torch.tensor([1]*len(steps))
        if "Wrong solution answer" in text:
            incorrect += torch.tensor([0]*(len(steps) - 1) + [1])

    labels = incorrect.tolist()
    labels = ["+" if i == 0 else "-" for i in labels]

    data_point = {
        "input": f"<|start_header_id|>user<|end_header_id|>\n\n"+problem_text+f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "value": steps,
        "label": labels
        }
    data.append(data_point)

output_dir = "finetune_dataset"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "in_distribution_dataset.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"\n总共 {len(data)} 条数据已存储到 {output_file}")