import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(666)


class ImageCaptionDataset(Dataset):
    def __init__(self, jsonl_file, img_dir):
        self.data = self.load_data(jsonl_file)
        self.img_dir = img_dir
        
    def load_data(self, jsonl_file):
        data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        caption = item['text']
        img_path = os.path.join(self.img_dir, str(item['image']))
        image = Image.open(img_path).convert('RGB')

        
        labels = item['labels']
        if "not harmful" in labels:
            label = 0
        else:
            label = 1
        
        return caption, image, torch.tensor(label)
    
def custom_collate(batch):
    texts, images, labels = zip(*batch)
    
    
    images = list(images)
    texts = list(texts)
    
    
    labels = torch.stack(labels, dim=0)
    
    return texts, images, labels




def load_test_harmc(batch_size):
    all_dataset = ImageCaptionDataset('/data4/zxf_1/datasets/MEME_datasets/harmC/test.jsonl', '/data4/zxf_1/datasets/MEME_datasets/harmC/images')

    
    
    test_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    return test_loader

def load_train_harmc(batch_size):
    all_dataset = ImageCaptionDataset('/data4/zxf_1/datasets/MEME_datasets/harmC/train.jsonl', '/data4/zxf_1/datasets/MEME_datasets/harmC/images')

    
    
    test_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    return test_loader



if __name__ == "__main__":
    dataload = load_test_harmc(1)
    print(len(dataload))
    for x1, x2, x3 in dataload:
        print(x1)
        
        break