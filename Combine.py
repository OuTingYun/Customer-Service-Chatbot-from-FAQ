import pandas as pd

std_dict=dict()

def single_mix(w2v_dic,bert_dic):
    buf_dic=dict() #standard
    buf_dic=w2v_dic
    first=list(bert_dic)[0]
    last=list(bert_dic)[-1]
    if bert_dic[first]>0.75 and bert_dic[last]<-0.95:
        buf_dic=bert_dic
    return buf_dic

def single_output(dic):
    first=list(dic)[0]
    print("Answer: ",std_dict[first])

def single_run(w2v_dic,bert_dic):
    dic=single_mix(w2v_dic,bert_dic)
    single_output (dic)

def readdata():
    data=pd.read_excel(".\\QA\\Total_User_Q.xlsx",engine='openpyxl',header=None, sheet_name='Std_Q_All', skiprows=[0])
    data.columns=["questions","answers"]
    for i in range (len (data)):
        std_dict[data['questions'][i]]=data['answers'][i]
readdata()
