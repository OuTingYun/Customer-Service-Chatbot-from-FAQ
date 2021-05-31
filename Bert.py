# Install library
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import re
from transformers import BertTokenizer
from transformers import  TFBertForSequenceClassification
from transformers import glue_convert_examples_to_features, InputExample

MAX_SEQ_LENGTH = 128
# Function
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

def example_to_features_predict(input_ids, attention_masks, token_type_ids):
    return {"input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids}

def example_to_features(input_ids, attention_masks, token_type_ids, y):
    return {"input_ids": input_ids,
         "attention_mask": attention_masks,
         "token_type_ids": token_type_ids},y

def get_prediction(data , tokenizer , model , in_sentences):
    label_list = data.label.values
    in_sentences[0]=re.sub(r'[^\w]', ' ', in_sentences[0])
    input_examples = [InputExample(guid="", text_a = in_sentences[0], text_b = None, label = 0)] 
    predict_input_fn = glue_convert_examples_to_features(examples=input_examples, tokenizer=tokenizer, max_length=MAX_SEQ_LENGTH, task='mrpc', label_list=label_list)
    x_test_input, y_test_input = transfer(predict_input_fn)
    test_ds   = tf.data.Dataset.from_tensor_slices((x_test_input[0], x_test_input[1], x_test_input[2])).map(example_to_features_predict).batch(32)

    predictions = model.predict(test_ds)
    #print(predictions)
    predictions_classes = np.argmax(predictions[0], axis = 1)

    #label rank
    arr = np.array(predictions[0][0])
    top_k_idx=arr.argsort()[::-1][0::]       
    score_dic={}

    #score rank
    arr2 = np.array(predictions[0][0])
    arr2.sort()
    arr2=arr2[::-1]
    #out_list=[]
    for j in range(len(top_k_idx)):
        score_dic[data['questions'][top_k_idx[j]]]=arr2[j]
    return score_dic

def bert_model():
    model = TFBertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels = 52)
    model.load_weights('.\\Model\\bert_model.h5')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    return model,tokenizer

def data_label(data):
    data['label']=""
    for i in range(len(data)):
        data['label'][i]=int(i)
    data['label']=data['label'].astype("int")
    return data