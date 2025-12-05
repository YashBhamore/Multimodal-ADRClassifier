import os
import pandas as pd
from brat_parser import get_entities_relations_attributes_groups

def parse_meddra_ann(path):
    """Parse MedDRA .ann format: TT1 10013649 9 19 text"""
    entities = []
    if not os.path.exists(path):
        return entities
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            ent_id = parts[0]
            meta = parts[1].split()
            if len(meta) < 3:
                continue
            code = meta[0]
            start, end = int(meta[1]), int(meta[2])
            text = parts[2]
            entities.append({
                "entity_id": ent_id,
                "entity_type": "MedDRA",
                "concept_id": code,
                "start_offset": start,
                "end_offset": end,
                "entity_text": text
            })
    return entities

def safe_readlines(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.readlines()
    except UnicodeDecodeError:
        with open(path, encoding="latin-1") as f:
            return f.readlines()
        
def parse_sct_ann(path):
    """Parse SCT .ann format: TT1 271782001 | Drowsy | 9 19 text"""
    entities = []
    if not os.path.exists(path):
        return entities
    
    try:
        lines = safe_readlines(path)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return entities
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ent_id = parts[0]
        meta = parts[1]
        try:
            # Example meta: "271782001 | Drowsy | 9 19"
            before_text = meta.strip().split("|")
            if len(before_text) >= 2:
                concept_id = before_text[0].strip()
                concept_name = before_text[1].strip()
            else:
                concept_id = meta.strip().split()[0]
                concept_name = "Unknown"
            # Extract last two numbers as offsets
            nums = [int(x) for x in meta.split() if x.isdigit()]
            start, end = nums[-2], nums[-1]
            text = parts[2]
            entities.append({
                # "entity_id": ent_id,
                # "entity_type": "SCT",
                # "label_id": concept_id,
                "label_name": concept_name,
                # "start_offset": start,
                # "end_offset": end,
                # "entity_text": text
            })
        except Exception as e:
            print(f"Skipping malformed line in {path}: {line} ({e})")
    return entities


def parse_original_ann(path):
    """Use brat_parser for the original BRAT format."""
    entities = []
    if not os.path.exists(path):
        return entities
    try:
        ents, _, _, _ = get_entities_relations_attributes_groups(path)
        for eid, e in ents.items():
            entities.append({
                "entity_id": eid,
                "entity_type": e.type,
                "start_offset": e.span[0][0],
                "end_offset": e.span[0][1],
                "entity_text": e.text
            })
    except Exception as e:
        print(f"Error parsing original: {path} ({e})")
    return entities

def load_cadec(cfg): 
    all_docs = []

    cadec_data_dir = os.path.join(cfg['data_dir'], 'cadec')
    text_dir = os.path.join(cadec_data_dir,'text')
    sct_dir = os.path.join(cadec_data_dir,'sct')

    for filename in os.listdir(text_dir):
        if not filename.endswith(".txt"):
            continue

        base = os.path.splitext(filename)[0]
        text_path = os.path.join(text_dir, filename)
        with open(text_path, encoding="utf-8") as f:
            text = f.read()

        doc_data = {
            "doc_id": base,
            "user_text": text,
            # "original": parse_original_ann(os.path.join(ORIGINAL_DIR, base + ".ann")),
            # "meddra": parse_meddra_ann(os.path.join(MEDDRA_DIR, base + ".ann")),
            "sct": parse_sct_ann(os.path.join(sct_dir, base + ".ann"))
        }
        all_docs.append(doc_data)

    df = pd.DataFrame(all_docs)

    df_exploded = df.explode("sct", ignore_index=True)

    # 2️⃣ Expand the dicts into columns
    sct_details = pd.json_normalize(df_exploded["sct"])

    # 3️⃣ Merge back with main DataFrame
    df_final = pd.concat([df_exploded.drop(columns=["sct"]), sct_details], axis=1)
    df_final = df_final[['user_text','label_name']]
    df_final = df_final.rename(columns={
                                        "user_text": "text",
                                        "label_name": "label",
                                    })

    out_path = os.path.join(cadec_data_dir,"cadec_dataset_grouped.csv")
    df_final.to_csv(out_path, index=False)
    print(f"\n✅ Saved dataset with {len(df_final)} grouped documents to {out_path}")
    
    return df_final
