
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms
from sklearn.metrics import f1_score, classification_report, accuracy_score

def api_metric(true_labels, pred_labels):
    # return f1_score(true_labels, pred_labels)
    print(classification_report(true_labels, pred_labels))
    return accuracy_score(true_labels, pred_labels)

def api_decode(outputs, labelvocab):
    formated_outputs = ['guid,tag']
    for guid, label in tqdm(outputs, desc='----- [Decoding]'):
        formated_outputs.append((str(guid) + ',' + labelvocab.id_to_label(label)))
    return formated_outputs

def api_encode(data, labelvocab, config):

    ''' 这里直接加入三个标签, 后面就不需要添加了 '''
    labelvocab.add_label('positive')
    labelvocab.add_label('neutral')
    labelvocab.add_label('negative')
    labelvocab.add_label('null')    # 空标签

    ''' 文本处理 BERT的tokenizer '''
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name)

    ''' 图像处理 torchvision的transforms '''
    def get_resize(image_size):
        for i in range(20):
            if 2**i >= image_size:
                return 2**i
        return image_size

    img_transform = transforms.Compose([
                transforms.Resize(get_resize(config.image_size)),
                transforms.CenterCrop(config.image_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ''' 对读入的data进行预处理 '''
    guids, encoded_texts, encoded_imgs, encoded_labels = [], [], [], []
    for line in tqdm(data, desc='----- [Encoding]'):
        guid, text, img, label = line
        # id
        guids.append(guid)
        
        # 文本
        text.replace('#', '')
        tokens = tokenizer.tokenize('[CLS]' + text + '[SEP]')
        encoded_texts.append(tokenizer.convert_tokens_to_ids(tokens))

        # 图像
        encoded_imgs.append(img_transform(img))
        
        # 标签
        encoded_labels.append(labelvocab.label_to_id(label))

    return guids, encoded_texts, encoded_imgs, encoded_labels

class APIDataset(Dataset):

    def __init__(self, guids, texts, imgs, labels) -> None:
        self.guids = guids
        self.texts = texts
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.guids)

    def __getitem__(self, index):
        return self.guids[index], self.texts[index], \
               self.imgs[index], self.labels[index]
    
    # collate_fn = None
    def collate_fn(self, batch):
        guids = [b[0] for b in batch]
        texts = [torch.LongTensor(b[1]) for b in batch]
        imgs = torch.FloatTensor([np.array(b[2]).tolist() for b in batch]) 
        labels = torch.LongTensor([b[3] for b in batch])

        ''' 处理文本 统一长度 增加mask tensor '''
        texts_mask = [torch.ones_like(text) for text in texts]

        paded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        paded_texts_mask = pad_sequence(texts_mask, batch_first=True, padding_value=0).gt(0)
        
        
        ''' 处理图像 '''

        return guids, paded_texts, paded_texts_mask, imgs, labels

class LabelVocab:
    UNK = 'UNK'

    def __init__(self) -> None:
        self.label2id = {}
        self.id2label = {}

    def __len__(self):
        return len(self.label2id)

    def add_label(self, label):
        if label not in self.label2id:
            self.label2id.update({label: len(self.label2id)})
            self.id2label.update({len(self.id2label): label})

    def label_to_id(self, label):
        return self.label2id.get(label)
    
    def id_to_label(self, id):
        return self.id2label.get(id)


class Processor:

    def __init__(self, config) -> None:
        self.config = config
        self.labelvocab = LabelVocab()
        pass

    def __call__(self, data, params):
        return self.to_loader(data, params)

    def encode(self, data):
        return api_encode(data, self.labelvocab, self.config)
    
    def decode(self, outputs):
        return api_decode(outputs, self.labelvocab)

    def metric(self, inputs, outputs):
        return api_metric(inputs, outputs)
    
    def to_dataset(self, data):
        dataset_inputs = self.encode(data)
        return APIDataset(*dataset_inputs)

    def to_loader(self, data, params):
        dataset = self.to_dataset(data)
        return DataLoader(dataset=dataset, **params, collate_fn=dataset.collate_fn)