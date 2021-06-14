## Customer-Service-Chatbot-from-FAQ

本專題製作一個能從常見問題集 (FAQ) 自動產生客服對話機器人的工具：使用者輸入問題 UserQ 後，程式將判斷 UserQ 屬於常見問題集中的哪個問題，並回傳相應的答案。

此程式使用三種方法判斷 UserQ 屬於常見問題集中的哪個問題，分別為
1. Bert
2. W2V + ElasticSearch
3. 結合 Bert 和 ( W2V+ElasticSearch ) 兩種方法

### Required Python libraries and versions

```python
elasticsearch = 7.10.2
tensorflow    = 2.4.1
transformer   = 4.5.0
numpy         = 1.19.5
regex         = 2.2.1
pandas        = 1.2.3
```

### A demo based on [國立中央大學計算機中心常見問題集(FAQ)](https://www.cc.ncu.edu.tw/page/qna)

下載 Customer-Service-Chatbot-from-FAQ.zip

在 Customer-Service-Chatbot-from-FAQ資料夾內開啟 cmd

並執行以下指令

```bash
#Use Bert
python Main.py -b [UserQ]

#Use W2V+ElasticSearch 
python Main.py -w [UserQ]

#Use Bert+(W2V+ElasticSearch)
python Main.py -c [UserQ]
```

### Train bert model according to different FAQ set

將問題集依照 `QA\Total_User_Q.xlsx` 內名為 `Std_Q_All`的工作表排列，並在`A1`填入問題數量。

![imgs](https://github.com/OuTingYun/Customer-Service-Chatbot-from-FAQ/blob/master/README/Std_Q_All.png)

填完後請執行以下指令

格式
```bash
python bert_train.py -fp <file of the standard questions>
                     -sn <whether you have the sheet name of the file>
                     -mp <the path for the model to store>

```
範例
```bash
python bert_train.py -fp C:\Users\user\Desktop\Customer-Service-Chatbot-from-FAQ\QA\Total_User_Q.xlsx 
                     -sn Std_Q_All 
                     -mp C:\Users\user\Desktop\Customer-Service-Chatbot-from-FAQ\Model\bert.h5
```


### Authors

Guan Ling Chou 周冠玲 

Ou Ting Yun    歐亭昀

### Reference

https://github.com/zake7749/word2vec-tutorial/blob/master/demo.py

https://sci2lab.github.io/ml_tutorial/bert_farsi_sentiment/
