import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader

IMG_SIZE=224
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR), # BILINEAR is common
    transforms.ToTensor(), # Converts to [C, H, W] and scales to [0.0, 1.0]
    transforms.Normalize(mean=mean, std=std)
])

def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)

    data_list=data['questions']
    image_path_list=[]
    questions=[]
    answers=[]
    for i in tqdm(range(len(data_list))):
        image_path=data_list[i]['image_filename']
        q=data_list[i]['question']
        a=data_list[i]['answer']
        questions.append(q)
        answers.append(a)
        image_path_list.append(image_path)
    return image_path_list, questions, answers
def load_image(local_path, split):
    if split not in ['trainA', 'testA', 'valA', 'testB', 'valB']:
        print("split must be one of train, test or val")
        return
    path=os.path.join(f'images/{split}',local_path )
    img = Image.open(path).convert('RGB')
    # scale=224/min(img.size[0], img.size[1])
    # img=img.resize(int(scale*img.size[0], scale*img.size[1]))
    # img = img.resize((224,224), Image.Resampling.NEAREST)
    # img = img.convert('RGB')
    # img_array = torch.tensor(np.array(img)).permute(2,0,1)
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    # std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    # img_array = (img_array - mean) / std
    img_tensor = image_transform(img)
    return img_tensor
class AnswerTokenzier():
    def __init__(self, answer):
        self.answer = answer
        self.vocab= None
        self.vocab_size = None
        self.pad_token = 0
        self.alphas={}
    def get_alpha_tensor(self):
        t=[]
        for i in self.alphas.keys():
            t.append(self.alphas[i])
        
        freqs=torch.tensor(t)

        freqs=freqs/torch.sum(freqs)
        inv_freq=1/freqs
        return inv_freq
        

    def build_vocab(self):
        vocab=[]
        for i in self.answer:
            if i not in vocab:
                vocab.append(i)
                self.alphas[i]=1
            else:
                self.alphas[i]+=1
                # continue
        vocab.append('<unk>')
        self.alphas['<unk>']=1
        self.vocab=vocab
        self.vocab_size=len(vocab)
    def tokenzie(self, text):
        tokens = text.split()
        # print(tokens)
        if len(tokens)>0:
            tokenized_text=self.vocab.index(tokens[0])
        else:
            tokenized_text=self.vocab.index('<unk>')
        return tokenized_text
    def decode(self, index):
        if index>=len(self.vocab):
            return '<unk>'
        else:
            return self.vocab[index]
    def batch_decode(self, indices):
        decoded_text = []
        for index in indices:
            decoded_text.append(self.vocab[index])
        return decoded_text
class ImageQADataset(Dataset):
    def __init__(self, images, questions, answers, split, bert_tokenizer, answer_tokenizer, max_question_len=64):
        self.images = images
        self.questions = questions
        self.answers = answers
        self.split = split
        self.bert_tokenizer = bert_tokenizer 
        self.answer_tokenizer = answer_tokenizer
        self.max_question_len = max_question_len

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_filename = self.images[index]
        # Assumes load_image (from data.py) can resolve the full image path 
        # using img_filename (e.g., "CLEVR_trainA_000000.png") and self.split (e.g., "trainA").
        image = load_image(img_filename, self.split)
        
        question_tokenized = self.bert_tokenizer(self.questions[index], padding='max_length', truncation=True, max_length=self.max_question_len, return_tensors="pt")
        question_ids = question_tokenized['input_ids'].squeeze(0)
        
        # answer_tokenizer.tokenzie should return a tensor (e.g., class index)
        answer = self.answer_tokenizer.tokenzie(self.answers[index])
        return image, question_ids, answer  
#  def get_max_question_length():
#     pass

# img, qna=load_data('/home/scai/msr/aiy247541/scratch/CLEVR_v1.0/questions/CLEVR_val_questions.json')
# print(img[0], qna[0])
# print(len(qna), len(img))
# print(load_image(img[0], 'val').shape)