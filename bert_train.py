import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
import re
from transformers import BertTokenizer
from transformers import  TFBertForSequenceClassification
from transformers import glue_convert_examples_to_features, InputExample
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import argparse

MAX_SEQ_LENGTH = 128
model_savepath=""

def load_data(path,sheet=None):
    data=pd.read_excel(path,header=None, sheet_name=sheet, skiprows=[0])
    data.columns=["questions","answers"]
    data['label']=""
    for i in range(len(data)):
        data['label'][i]=int(i)
    data['questions']=data['questions'].astype("str")
    data['answers']=data['answers'].astype("str")
    data['label']=data['label'].astype("int")
    for i in range(len(data)):
        data['questions'][i]=re.sub(r'[^\w]', ' ', data['questions'][i])
    return data

def convert_data_into_input_example(data):
    input_examples = []
    for i in range(len(data)):
        example = InputExample(
            guid= None,
            text_a= data.iloc[i]['questions'],
            text_b= None,
            label= data.iloc[i]['label']
        )
        input_examples.append(example)
    return input_examples

def transfer(bdset):
    input_ids, attention_mask, token_type_ids, label = [], [], [], []
    for in_ex in bdset:
        input_ids.append(in_ex.input_ids)
        attention_mask.append(in_ex.attention_mask)
        token_type_ids.append(in_ex.token_type_ids)
        label.append(in_ex.label)

    input_ids = np.vstack(input_ids)
    attention_mask = np.vstack(attention_mask)
    token_type_ids = np.vstack(token_type_ids)
    label = np.vstack(label)
    return ([input_ids, attention_mask, token_type_ids], label)

def example_to_features(input_ids, attention_masks, token_type_ids, y):
    return {"input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids},y

def train(train_ds):
    EPOCHS = 10
    model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels = len(label_list))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08, clipnorm=1.0)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    model.fit(train_ds, epochs=EPOCHS)
    my_callbacks = [
        EarlyStopping(monitor='accuracy',patience=5, mode='max',verbose=1,),
        ModelCheckpoint(model_savepath, monitor='accuracy', patience=5, verbose=1, save_best_only=True, mode='max', period=1),
    ]
    model.fit(train_ds, epochs=50, callbacks=my_callbacks)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.description='please enter filepath and model path'
    parser.add_argument("-fp","--file_path", required=True,help="filepath of the standard question",type=str)
    parser.add_argument("-sn","--sheet_name",help="sheet name of the standard question",type=str)
    parser.add_argument("-mp","--model_path", required=True,help="path to store the model",type=str)
    args=parser.parse_args()
    model_savepath=args.model_path

    #path="D:\\專題(3下)\\code\\project\\QA\\Total_User_Q.xlsx"
    #sheet="Std_Q_All"

    #load train data
    data=load_data(args.file_path,args.sheet_name)
    label_list = data.label.values
    print("Training ....")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    train_input_examples = convert_data_into_input_example(data)
    bert_train_dataset = glue_convert_examples_to_features(examples=train_input_examples, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH, task='mrpc', label_list=label_list)
    x_train, y_train = transfer(bert_train_dataset)
    train_ds = tf.data.Dataset.from_tensor_slices((x_train[0], x_train[1], x_train[2], y_train)).map(example_to_features).shuffle(100).batch(32)
    train(train_ds)
    