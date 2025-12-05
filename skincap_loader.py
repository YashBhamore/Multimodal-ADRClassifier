from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os
from tqdm import tqdm
import requests
import pandas as pd

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

def load_skincap(cfg):
    # # Login using e.g. `huggingface-cli login` to access this dataset
    # dataset = load_dataset("joshuachou/SkinCAP", streaming=True)

    save_dir = os.path.join(cfg['data_dir'], 'skincap')
    # os.makedirs(save_dir, exist_ok=True)

    # for i, sample in enumerate(tqdm(dataset["train"])):
    #     img = sample["image"]
    #     img_path = os.path.join(save_dir, f"img_{i+1:05d}.png")
    #     img.save(img_path)
    #     # if i > 10:
    #     #     break

    url = "https://huggingface.co/datasets/joshuachou/SkinCAP/resolve/main/skincap_v240715.xlsx"
    out_path = os.path.join(cfg['data_dir'], "skincap_v240715.xlsx")

    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

    r = requests.get(url, headers=headers)
    r.raise_for_status()

    with open(out_path, "wb") as f:
        f.write(r.content)

    df = pd.read_excel(out_path, header=1)
    df = df[['caption_en','disease']]
    df["image_path"] = [os.path.join(save_dir, f"img_{i+1:05d}.png") for i in range(len(df))]

    df_final = df.rename(columns={
                                        "caption_en": "text",
                                        "disease": "label",
                                    })
    df_final.to_excel(out_path, index=False)
    print(f"Saved processed data to {out_path}")
    
    return df_final