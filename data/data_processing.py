import os 
import pandas as pd

#Path and list initialization

train_path = 'train/'
test_path = 'test/'
emotions = set()
train_emotions_paths = []
test_emotions_paths = []

for folder in os.listdir(train_path):
    if os.path.isdir(train_path+folder):
        emotions.add(folder)
        train_emotions_paths.append([(train_path+folder),folder])
    
for folder in os.listdir(test_path):
    if os.path.isdir(test_path+folder):
        emotions.add(folder)
        test_emotions_paths.append([(test_path+folder),folder])

emotions = list(emotions)
emotions_dict = dict(zip(emotions,range(len(emotions))))


#Dataset Construction
train_rows = []
for path,emotion in train_emotions_paths:
    for image in os.listdir(path):
        image_dict = {}
        image_path = 'data/'+path+'/'+image
        image_dict['path'] = image_path
        image_dict['label'] = emotions_dict[emotion]
        train_rows.append(image_dict)

train_df = pd.DataFrame(train_rows)
pd.DataFrame.to_csv(train_df,'train/train.csv',index=False)

test_rows = []
for path,emotion in test_emotions_paths:
    for image in os.listdir(path):
        image_dict = {}
        image_path = 'data/'+path+'/'+image
        image_dict['path'] = image_path
        image_dict['label'] = emotions_dict[emotion]
        test_rows.append(image_dict)

test_df = pd.DataFrame(test_rows)
pd.DataFrame.to_csv(test_df,'test/test.csv',index=False)
