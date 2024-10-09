import dill
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

def resize(mylist,length):
    if len(mylist) >= length:
        mylist = mylist[0:length]
    else:
        mylist.extend([0.0]*(length-len(mylist)))
    return mylist

def read_series(file,length):
    with open(file,mode='r',encoding='utf-8') as f:
        lines = list(f.readlines())
        temp_list = lines[0].strip().replace('[','').replace(']','').split(',')
        timestamps = [float(i) for i in temp_list if i!='']
        first = timestamps[0]
        timestamps = [i-first for i in timestamps]
        temp_list = lines[3].strip().replace('[','').replace(']','').split(',')
        lengths = [abs(float(i)) for i in temp_list if i!='']
        temp_list = lines[4].strip().replace('[','').replace(']','').split(',')
        s2c_lengths = [abs(float(i)) for i in temp_list if i!='']
        temp_list = lines[5].strip().replace('[','').replace(']','').split(',')
        c2s_lengths = [abs(float(i)) for i in temp_list if i!='']
    timestamps = resize(timestamps,length)
    lengths = resize(lengths,length)
    s2c_lengths = resize(s2c_lengths,length)
    c2s_lengths = resize(c2s_lengths,length)
    if (timestamps is None) or (lengths is None):
        timestamps = [0.0] * length
        lengths = [0.0] * length
    if s2c_lengths is None:
        s2c_lengths = [0.0] * length
    if c2s_lengths is None:
        c2s_lengths = [0.0] * length
    res = np.array([timestamps,lengths,s2c_lengths,c2s_lengths],dtype='float32').T
    return res

def traverse_image_folder(folder_path,text_path):
    labels = []
    images = []
    num_series = []
    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)
        if os.path.isdir(category_path):
            for image_file in os.listdir(category_path):
                image_path = os.path.join(category_path, image_file)
                image = Image.open(image_path)
                image = image.resize((80, 80))
                try:
                    traffic_series = read_series(os.path.join(text_path,image_file.replace('.png','.txt')),128)
                except:
                    continue
                if image is not None:
                    labels.append(category)
                    images.append(image)
                    num_series.append(traffic_series)
    return labels, images, num_series

# sequences,labels = read_dataset(text_path,length)
text_path = './twitterbot_seq/series'
train_img_path = './0924_bot_png/train'
test_img_path = './0924_bot_png/test'

class RumorDataset(Dataset):
    def __init__(self, labels,images,texts):
        # self.data_dir = data_dir
        self.labels = labels
        self.images = images
        self.texts = texts

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.images[index]
        text = self.texts[index]
        label_dict={'amplifier':0,'auto-response':1,'hybrid':2,'publish':3,'report':4}
        img = transforms.ToTensor()(img)
        text = torch.as_tensor(text)
        label = torch.as_tensor(label_dict[label])
        return img,text,label
 
    def __len__(self):
        return len(self.labels)

test_label,test_image,test_series = traverse_image_folder(test_img_path,text_path)
test_dataset = RumorDataset(labels = test_label, images = test_image,texts = test_series)
print(len(test_dataset))
train_label,train_image,train_series = traverse_image_folder(train_img_path,text_path)
train_dataset = RumorDataset(labels = train_label, images = train_image,texts = train_series)
print(len(train_dataset))
with open('./dataset_train80.pkl','wb') as f:
    dill.dump(train_dataset, f)

with open('./dataset_test80.pkl','wb') as f:
    dill.dump(test_dataset, f)
