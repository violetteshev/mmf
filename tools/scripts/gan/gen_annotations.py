import os
import json
import numpy as np
from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.file_io import PathManager


def get_imdb(ann_path: str, ans_path: str) -> np.ndarray:

    imdb = [{"dataset_name": "gan_vqa2"}]
    old_imdb = np.load(ann_path, allow_pickle=True)

    with PathManager.open(ans_path, "r") as f:
        answers = json.load(f)

    answers_dict = {}
    for ans in answers:
        q_id = ans["question_id"]
        answer = ans["answer"]
        answers_dict[q_id] = answer

    unans_count = 0
    for item in old_imdb[1:]:
        q_id = item["question_id"]

        if not q_id in answers_dict:
            continue

        pred_answers = answers_dict[q_id]
        sorted_ans = [s[0] for s in sorted(pred_answers.items(), key=lambda x: x[1], reverse=True)]
        item["pred_answers"] = sorted_ans
        item["correct_answers"] = []

        for ans in sorted_ans:
            if ans in item["all_answers"]:
                item["correct_answers"].append(ans)
        
        if len(item["correct_answers"]) == 0:
            item["correct_answers"].append("unanswerable")
            unans_count += 1
        
        imdb.append(item)

    print(f"Unanswerable questions: {unans_count} out of {len(imdb)-1}")
    return np.array(imdb)


if __name__ == "__main__":
    split = "nominival"
    root_dir = os.path.join(get_mmf_cache_dir(), "data", "datasets", "vqa2", "defaults")
    ann_path = os.path.join(root_dir, "annotations", f"imdb_val2014.npy")
    ans_path = os.path.join("../LXMERT_original/snap/vqa/base", f"{split}_predict_topk.json")
    res_path = os.path.join(root_dir, "annotations", f"imdb_gan_{split}.npy")

    imdb = get_imdb(ann_path, ans_path)

    np.save(res_path, imdb)

