import sys
import os
import json
import numpy as np
from tqdm import tqdm
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.text import tokenize
from mmf.utils.file_io import PathManager


def get_imdb(ann_path: str, quest_path: str, split: str, answer_vocab_path: str) -> np.ndarray:

    imdb = [{"dataset_name": "okvqa"}]

    with PathManager.open(answer_vocab_path, "r") as f:
        answer_vocab = set(f.read().splitlines())

    with PathManager.open(ann_path, "r") as f:
        annotations = json.load(f)["annotations"]
    
    with PathManager.open(quest_path, "r") as f:
        questions = json.load(f)["questions"]

    gt_answers = {}
    for ann in annotations:
        gt_answers[ann["question_id"]] = ann["answers"]
    
    count = 0
    for quest in tqdm(questions):
        image_name = f"COCO_{split}_{quest['image_id']:012d}"
        q_id = quest["question_id"]
        all_answers = [item['answer'] for item in gt_answers[q_id]]
        answers = [ans for ans in all_answers if ans in answer_vocab]

        if len(answers) == 0:
            answers = ["<unk>"]
            count += 1
    
        entry = {
            "image_name": image_name,
            "image_id": quest["image_id"],
            "feature_path": f"{image_name}.npy",
            "question_id": q_id,
            "question_str": quest["question"],
            "question_tokens": tokenize(quest["question"]),
            "answers": answers,
            "all_answers": all_answers,
        }

        imdb.append(entry)
    print("Unknown questions:", count)

    return np.array(imdb)


if __name__ == "__main__":
    split = "val2014"
    root_dir = os.path.join(get_mmf_cache_dir(), "data", "datasets", "okvqa", "defaults")
    ann_path = os.path.join(root_dir, "annotations", f"mscoco_{split}_annotations.json")
    quest_path = os.path.join(root_dir, "annotations", f"OpenEnded_mscoco_{split}_questions.json")
    answer_vocab_path = os.path.join(root_dir, "extras", "vocabs", "answers_okvqa.txt")
    res_path = os.path.join(root_dir, "annotations", f"imdb_{split}.npy")

    imdb = get_imdb(ann_path, quest_path, split, answer_vocab_path)

    np.save(res_path, imdb)
