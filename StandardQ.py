from elasticsearch import Elasticsearch
from openpyxl import load_workbook
import json
es = Elasticsearch()
import warnings
warnings.filterwarnings("ignore")

'''
將所有標準答案放入elasticsearch裡面
'''
def StdQ_to_Elastic(Path):
    wb = load_workbook(Path)
    sheet=wb['Std_Q_All']
    Q=sheet['A']
    A=sheet['B']
    num=int(Q[0].value)
    question=[]
    for i in range (1,num+1):
        question.append(Q[i].value)
        data = {'title': Q[i].value, 'Ans': A[i].value}
        result = es.create(index='curpus2', doc_type='politics', id=i, body=data)
        result = es.get(index="curpus2", doc_type="politics", id=i)
        j=json.dumps(result,separators=(',\n', ': '), ensure_ascii=False).encode('utf8')
def Put_All_StdQ_to_Els():
    es.indices.delete(index='curpus2', ignore=[400, 404])
    Std_place='.\QA\\Total_User_Q.xlsx'
    StdQ_to_Elastic(Std_place)
