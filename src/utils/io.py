from typing import List, Dict
import os
import json

def save_checkpoint(
    ckpt_folder: str,
    img_paths: List[str],
    gpt_raw_answers: List[str],
):
    content = [
        {img_path: gpt_raw_answer for img_path, gpt_raw_answer in zip(img_paths, gpt_raw_answers)}
    ]
    
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
    with open(os.path.join(ckpt_folder, "ckpt.json"), "w") as f:
        json.dump(content, f)
        
def load_checkpoint(
    ckpt_path: str,  
) -> List[Dict]:
    with open(ckpt_path, "r") as f:
        content_list = json.load(f)
    return content_list