import warnings

warnings.filterwarnings("ignore")

import argparse
import Combine as cb

parser = argparse.ArgumentParser()
parser.description='please enter user query'
parser.add_argument("-w","--w2v",help="predict user query in W2V+ElasticSearch",type=str)
parser.add_argument("-b","--bert",help="predict user query in Bert",type=str)
parser.add_argument("-c","--combine",help="predict user query in  the combination of W2V+ElasticSearch and Bert",type=str)

args=parser.parse_args()

print("Running...")
import BertW2V as bertw2v
import StandardQ as SQ

if args.w2v:
    # 將所有標準問題放入ElasticSearch
    SQ.Put_All_StdQ_to_Els()
    w2v_dic=bertw2v.RunW2V(args.w2v)
    cb.single_output(w2v_dic)
if args.bert:
    bert_dic=bertw2v.RunBert(args.bert)
    cb.single_output(bert_dic)
if args.combine:
    SQ.Put_All_StdQ_to_Els()
    bert_dic,w2v_dic=bertw2v.Run(args.combine)
    cb.single_run(w2v_dic,bert_dic)

