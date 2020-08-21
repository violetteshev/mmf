import os
import json
from tqdm import tqdm

from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.file_io import PathManager
from mmf.datasets.processors.processors import EvalAIAnswerProcessor


root_dir = os.path.join(get_mmf_cache_dir(), "data", "datasets", "okvqa", "defaults", "annotations")
out_dir = os.path.join(get_mmf_cache_dir(), "data", "datasets", "okvqa", "defaults", "extras", "vocabs")
train_path = os.path.join(root_dir, "mscoco_train2014_annotations.json")
val_path = os.path.join(root_dir, "mscoco_val2014_annotations.json")
out_path = os.path.join(out_dir, "gt2raw_answers.json")

evalai_answer_processor = EvalAIAnswerProcessor()

with PathManager.open(train_path, "r") as f:
    annotations = json.load(f)["annotations"]

with PathManager.open(val_path, "r") as f:
    annotations += json.load(f)["annotations"]

gt2raw = {}
for ann in tqdm(annotations):
    for ans in ann["answers"]:
        raw_ans = evalai_answer_processor(ans["raw_answer"])
        gt_ans = evalai_answer_processor(ans["answer"])

        if gt_ans in gt2raw:
            gt2raw[gt_ans].add(raw_ans)
        else:
            gt2raw[gt_ans] = set([raw_ans])

gt2raw = {k: list(v) for k, v in gt2raw.items()}
with PathManager.open(out_path, "w") as f:
    json.dump(gt2raw, f, indent=4)
