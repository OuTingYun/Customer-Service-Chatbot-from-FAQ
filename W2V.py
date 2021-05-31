""" 載入model"""
import warnings

warnings.filterwarnings("ignore")
from gensim.models import word2vec
from gensim import models
import jieba
from elasticsearch import Elasticsearch
import json
import  numpy
'''跑單一個使用者問題'''
def Run_Single_Query(query,Model):
    Total_dic=dict()
    Keyword=list()
    QA=list()
    # 載入model
    """ 用jieba切割使用者問句 """
    def Split_UserQ():
        q_list = query.split()
        seg_list=[]
        seg_list = jieba.cut(query, cut_all=False)
        seg_list="$".join(seg_list).split("$")
        return seg_list
    #切割使用者問題
    Seg_List=Split_UserQ()

    """ 將所有切割後的片段放進model裡面，找出所有相似詞 """
    def Seg_Similar():
        L=[]
        Error=[]
        for item in Seg_List:
            item_list=[]
            item_list.append(item)
            try: 
                res = Model.wv.most_similar(item,topn = 3)
                # res是相似詞
                for itemres in res:
                    item_list.append(itemres[0])
            except Exception as e:
                Error.append(repr(e))
            L.append(item_list)
        return L
    
    #尋找切割後的詞的相似詞
    Keyword=Seg_Similar()

    #所有的詞拿去match
    for items in Keyword:
        numpy.unique(items).tolist()
        Total_dic=Query(items[0],3,Total_dic) #將原本文字的比重*3倍
        if(len(items)>2):
            for i in range(1,len(items)):
                Total_dic=Query(items[i],1,Total_dic)
    #按照分數做sort
    Total_dic2=sorted(Total_dic.items(), key=lambda x:x[1],reverse=True)
    Total_Dic=dict()
    for item in Total_dic2:
        Total_Dic[item[0]]=item[1]
    return Total_Dic

"""找出match放入的字串的標準達案"""
def Query(Q,rate,Total_dic):
    es = Elasticsearch()
    dsl = { 'query' : { 'match' :{ 'title' : Q}}}
    result = es.search(index = 'curpus2' , doc_type = 'politics',body=dsl)
    for i in range(len(result['hits']['hits'])):
        QA_text=result['hits']['hits'][i]['_source']['title']
        QA_score=rate*result['hits']['hits'][i]['_score']
        if Total_dic.get(QA_text,True)==True:
            Total_dic[QA_text]=QA_score
        else:
            Total_dic[QA_text]=Total_dic[QA_text]+QA_score
        j=json.dumps(result,separators=(',\n', ': '), ensure_ascii=False).encode('utf8')
    return Total_dic