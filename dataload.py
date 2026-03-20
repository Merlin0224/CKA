import spacy
import random
import io
import os
from PIL import Image
import pyarrow.dataset as ds
import glob
import numpy as np

def build_pope_dataset(data_dir, num_samples=50):

    parquet_files = glob.glob(os.path.join(data_dir, "*.parquet"))
    print(f"✅ 在目录 {data_dir} 中发现 {len(parquet_files)} 个数据分片。")
    
    dataset = ds.dataset(parquet_files, format="parquet")
    
    all_data = dataset.to_table().to_pandas()
    print(f"✅ 数据加载完毕，总条目数: {len(all_data)}")
    
    pope_data = []
    # 随机取 100 个下标用于后续的正负样本构造
    indices = random.sample(range(len(all_data)), num_samples * 2)
    
    for i in range(num_samples):
        # 取第 i 条作为正样本
        row_pos = all_data.iloc[indices[i]]
        img_bytes = row_pos['image']['bytes'] if isinstance(row_pos['image'], dict) else row_pos['image']
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # 提取名词
        caption_raw = row_pos['caption']
        if isinstance(caption_raw, (list, np.ndarray)):
            # 如果是列表/数组，取第一个元素，并确保转为 str
            caption = str(caption_raw[0])
        else:
            caption = str(caption_raw)
        obj_pos = extract_object_from_caption(caption)
        
        # 构造正样本
        pope_data.append({
            "image": image,
            "prompt": f"Is there a {obj_pos} in the image? Answer Yes or No.",
            "gt": True
        })
        
        # 构造负样本
        row_neg = all_data.iloc[indices[i + num_samples]]
        caption_neg = row_neg['caption'][0] if isinstance(row_neg['caption'], list) else row_neg['caption']
        obj_neg = extract_object_from_caption(caption_neg)
        
        if obj_neg != obj_pos:
            pope_data.append({
                "image": image,
                "prompt": f"Is there a {obj_neg} in the image? Answer Yes or No.",
                "gt": False
            })
            
    return pope_data


nlp = spacy.load("en_core_web_sm")
def extract_object_from_caption(caption):
    """
    通过 NLP 词性标注，自动提取 caption 中的名词 (NOUN)
    """
    # 强制类型转换：如果是数组/列表，取第一个元素，并强转为字符串
    if isinstance(caption, (list, np.ndarray)):
        caption_str = str(caption[0])
    else:
        caption_str = str(caption)
        
    doc = nlp(caption_str)
    
    # 提取 NOUN
    nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN"]
    
    # 过滤掉无效词
    ignore_list = {'time', 'way', 'day', 'part', 'side', 'background', 'place', 'moment'}
    valid_nouns = [n for n in nouns if n not in ignore_list]
    
    return valid_nouns[0] if valid_nouns else "person"