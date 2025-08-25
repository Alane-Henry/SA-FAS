import json
import torch
import os
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

lable_dict = {'0_0_0': 'Live Face',
                '1_0_0': 'Print',
                '1_0_1': 'Replay',
                '1_0_2': 'Cutouts',
                '1_1_0': 'Transparent',
                '1_1_1': 'Plaster',
                '1_1_2': 'Resin',
                '2_0_0': 'Attribute-Edit',
                '2_0_1': 'Face-Swap',
                '2_0_2': 'Video-Driven',
                '2_1_0': 'Pixcel-Level',
                '2_1_1': 'Semantic-Leve',
                '2_2_0': 'ID_Consisnt',
                '2_2_1': 'Style',
                '2_2_2': 'Prompt',
              }

json_path = 'prompts4.json'
with open(json_path, 'r') as f:
    prompt_dict = json.load(f)

# 用于存储文本特征
text_features_by_class = {}

# 每个类别分别处理
for label, prompts in prompt_dict.items():
    inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)  # 归一化
    text_features_by_class[label] = features.cpu()  # 移回 CPU，便于保存

# 保存为 .pt 文件（可选）
torch.save(text_features_by_class, "text_features_by_class4.pt")