## Customer-Service-Chatbot-from-FAQ
本專題製作一個能從常見問題集 (FAQ) 自動產生客服對話機器人的工具：使用者輸入問題 User Q 後，程式將判斷User Q屬於常見問題集中的哪個問題，並回傳相應的答案。

此程式使用三種方法判斷UserQ屬於常見問題集中的哪個問題，分別為
1. Bert
2. W2V+ElasticSearch
3. 結合 Bert 和 (W2V+ElasticSearch) 兩種方法
### Environment
```python
elasticsearch = 7.10.2
tensorflow    = 2.4.1
transformer   = 4.5.0
numpy         = 1.19.5
regex         = 2.2.1
pandas        = 1.2.3
```
### How to use

下載Customer-Service-Chatbot-from-FAQ.zip

在 Customer-Service-Chatbot-from-FAQ資料夾內開啟cmd

並執行以下指令

```bash
#Use Bert
python Main.py -b [UserQ]

#Use W2V+ElasticSearch 
python Main.py -w [UserQ]

#Use Bert+(W2V+ElasticSearch)
python Main.py -c [UserQ]
```
### Reference

https://github.com/zake7749/word2vec-tutorial/blob/master/demo.py