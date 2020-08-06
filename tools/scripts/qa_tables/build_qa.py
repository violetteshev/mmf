import os
import json

from mmf.utils.configuration import get_mmf_cache_dir
from mmf.utils.file_io import PathManager


if __name__ == "__main__":
    src_dataset = 'vqa2'
    dst_dataset = 'okvqa'
    src_fname = "answers_vqa.txt"
    dst_fname = "answers_okvqa.txt"
    out_fname = f"{src_dataset}2{dst_dataset}.json"
    src_dir = os.path.join(get_mmf_cache_dir(), "data", "datasets", src_dataset, "defaults", "extras", "vocabs")
    dst_dir = os.path.join(get_mmf_cache_dir(), "data", "datasets", dst_dataset, "defaults", "extras", "vocabs")

    with PathManager.open(os.path.join(src_dir, src_fname), "r") as f:
        src_vocab = f.read().splitlines()
    
    with PathManager.open(os.path.join(dst_dir, dst_fname), "r") as f:
        dst_vocab = f.read().splitlines()
    
    src_dict = {w: i for i, w in enumerate(src_vocab)}
    qa_map = {}
    count = 0
    for idx, word in enumerate(dst_vocab):
        if word in src_dict:
            qa_map[idx] = src_dict[word]
            count += 1
    
    with PathManager.open(os.path.join(dst_dir, out_fname), "w") as f:
        json.dump(qa_map, f, indent=4)
    print("Found answers:", count)
