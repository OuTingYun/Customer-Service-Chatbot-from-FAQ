import W2V as w2v
import Bert as bert

def RunBert(UserQ):
    model,tokenizer=bert.bert_model()
    data=bert.pd.read_excel(".\QA\Total_User_Q.xlsx",engine='openpyxl',header=None, sheet_name='Std_Q_All', skiprows=[0])
    data.columns=["questions","answers"]
    data=bert.data_label(data)
    buf=list()
    buf.append(UserQ)
    bert_dic=bert.get_prediction(data , tokenizer , model , buf)
    return bert_dic

def RunW2V(UserQ):
    # 載入model
    Model = w2v.models.Word2Vec.load('.\Model\\W2V_Jieba.model')
    Total_dic=w2v.Run_Single_Query(UserQ,Model)
    return Total_dic

def Run(UserQ):
    bert_dic= RunBert(UserQ)
    w2v_dic = RunW2V(UserQ)
    return bert_dic,w2v_dic
