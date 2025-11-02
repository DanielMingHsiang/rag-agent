# 資料夾結構說明
- langgraph_react_agent - 使用 Langgraph 官方樣板改寫成 RAG Agent
- qdrant_insert_data.ipynb - 將文檔切塊、嵌入、insert 到向量資料庫
# 文檔素材
- [AWS 官方 EC2 文檔下 PDF](https://docs.aws.amazon.com/zh_tw/AWSEC2/latest/UserGuide/concepts.html)
# RAG 檢索示範影片 （查詢 EC2 設定方式）
![img](images/rag-agent-demo.gif)

# 技術棧
- 編排框架 [LangChain / Langgraph](https://docs.langchain.com/oss/python/langgraph/overview)
- 向量資料庫 [Qdrant](https://qdrant.tech/)
- Embedding 模型
    - 北京智源人工智能研究院 [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)
    - 微軟 [Multilingual E5 Large](https://huggingface.co/intfloat/multilingual-e5-large)
    - Cohere [Cohere-embed-multilingual-v3.0](https://huggingface.co/Cohere/Cohere-embed-multilingual-v3.0)
    - Google [gemini-embedding-exp-03-07](https://developers.googleblog.com/en/gemini-embedding-text-model-now-available-gemini-api/)
- 大語言模型 [claude-3-5-sonnet](https://www.anthropic.com/news/claude-3-5-sonnet)

# 嵌入模型繁體中文排行
- 這個實作使用的嵌入模型，是參考針對繁體中文支援度高的前幾名進行 RAG 向量嵌入與檢索測試
- 參考資料：[ihower 使用繁體中文評測各家 Embedding 模型的檢索能力](https://ihower.tw/blog/12167-embedding-models)
![嵌入模型繁體中文排行](https://ihower.tw/blog/wp-content/uploads/2024/07/embedding-model-zh-tw-benchmark.jpg)

# RAG 流程圖
在這個實作當中完整實作 RAG 兩大作業流程
- 檢索增強生成：提示詞輸入 -> 提示詞嵌入 -> 向量檢索 -> 增強生成
- 資料攝取：解析原始文檔 -> 文檔切塊 -> 向量嵌入 -> 向量資料庫
![img](images/rag-flow.png)

# 細節
- BAAI/bge-m3 嵌入模型使用「密集向量（語義相似度）」、「稀疏向量（關鍵字匹配）」達到混合檢索 hybrid search
    - 因測試集素材有限，使用混合檢索是否精準度更高，需要更多的評估才能測試，因爲這個實作目的在於完整實踐一遍整個 RAG 流程，因此不深入探討
- RAG 檢索精準度提高的方式可以透過「metadata 過濾」，例如檢索過程過明確濾掉不相關的文字分塊，提供給 LLM 與使用者提問更精準匹配的資訊

