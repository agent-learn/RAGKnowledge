import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 加速模型下载
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from rag_utils import generate_doc_id, clean_and_split_documents
from sentence_transformers import CrossEncoder
from langchain_classic.retrievers.document_compressors.cross_encoder import BaseCrossEncoder

os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))


# ================= 1. 初始化组件 =================
llm=ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)
embeddings=OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
)

# ================= 2. 数据库初始化（带持久化） =================
chroma_db_path="./enterprise_kb_db"
vectorstore=Chroma(
    collection_name="kb_collection",
    embedding_function=embeddings,
    persist_directory=chroma_db_path
)

# ================= 3. 数据导入（增量更新） =================
raw_docs=[
    Document(page_content="LangGraph 的 Checkpointer 机制用于持久化。",metadata={"category":"framework","year":2024}),
    Document(page_content="生产环境推荐使用 SqliteSaver 或 PostgresSaver。",metadata={"category":"framework","year":2024}),
    Document(page_content="Python 3.12 引入了新的类型参数语法。",metadata={"category":"language","year":2023}),
    Document(page_content="Art_1062 型号的技术参数是 500W 功率。",metadata={"category":"hardware","year":2024}),
    Document(page_content="公司年会定于 12 月 20 日在总部礼堂举行。",metadata={"category": "admin", "year": 2024}),
    Document(page_content="食堂本周菜单以川菜为主，周五供应水饺。",metadata={"category": "admin", "year": 2024}),
]

def ingest_data(docs):
    split_docs=clean_and_split_documents(docs)
    for doc in split_docs:
        doc.metadata["doc_id"]=generate_doc_id(doc.page_content)

    existing_ids=set(vectorstore.get()["ids"])
    new_docs,new_ids=[],[]
    for doc in split_docs:
        if doc.metadata["doc_id"] not in existing_ids:
            new_docs.append(doc)
            new_ids.append(doc.metadata["doc_id"])

    if new_docs:
        vectorstore.add_documents(new_docs,ids=new_ids)
        print(f"✅ 增量更新: 新增 {len(new_docs)} 个文本块。")
    else:
        print("🔄 数据无变化，跳过入库。")

ingest_data(raw_docs)   # 第一次运行会入库
# ingest_data(raw_docs) # 如果取消注释再跑一次，会显示“数据无变化”

# ================= 4. 构建终极检索器 =================
all_docs_for_bm25=[
    Document(page_content=t,metadata=m)
    for t,m in zip(vectorstore.get()["documents"],vectorstore.get()["metadatas"])
]
#权重
HYBRID_WEIGHTS=[0.4,0.6]

# 路 1：BM25
bm25=BM25Retriever.from_documents(all_docs_for_bm25,k=4)
# 路 2：向量
vector=vectorstore.as_retriever(search_kwargs={"k":4})
# 混合
hybrid=EnsembleRetriever(
    retrievers=[bm25,vector],
    weights=HYBRID_WEIGHTS
)

# 精排
#cross_encoder=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

class LocalCrossEncoder(BaseCrossEncoder):
    def __init__(self,model_name:str):
        self.client=CrossEncoder(model_name,local_files_only=True)
    def score(self,text_pairs):
        return self.client.predict(text_pairs)
    
cross_encoder=LocalCrossEncoder("BAAI/bge-reranker-v2-m3")
reranker=CrossEncoderReranker(model=cross_encoder,top_n=3)
final_retriever=ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=hybrid
)

def safe_retrieve(query:str):
    """精排 + 异常降级"""
    try:
        docs=final_retriever.invoke(query)
        if docs:
            return docs        #正常精排，返回结果
        print("⚠️ 精排返回空，降级为混合检索前 3 条")    #精排输出空，降级
        fallback = hybrid.invoke(query)[:3]
        if not fallback:
            print ("⚠️ 混合检索也为空，最终无可用文档")
        return fallback
    except Exception as e:
        print(f"⚠️ 精排异常，降级为混合检索: {e}")
        return hybrid.invoke(query)[:3]

# ================= 5. RAG 闭环生成 =================
def ask_enterprise(query:str):
    print(f"\n🙋‍♂️ 用户: {query}")

    docs=safe_retrieve(query)

    if not docs:  #空检索兜底，不调llm
        print("🤖 专家: 未检索到相关资料。")
        return

    context="\n".join([d.page_content for d in docs])

    prompt=f"""你是一个严谨的企业技术助手。请严格根据以下参考资料回答问题。
规则：
1. 请综合列出参考资料中所有与问题相关的要点，不要遗漏，并且与问题无关的不要回答。
2. 如果参考资料不足以回答问题，回答“根据现有资料无法确认”，不要编造。
3. 不要使用参考资料之外的知识。

参考资料：
{context}

问题：{query}
"""
    response=llm.invoke(prompt)
    print(f"🤖 专家: {response.content}")

DEBUG = False

def debug_retrieve(query: str):
    if not DEBUG:
        return
    bm25_docs = bm25.invoke(query)
    vector_docs = vector.invoke(query)
    hybrid_docs = hybrid.invoke(query)
    final_docs = safe_retrieve(query)
    fmt = lambda ds: [d.page_content[:24] for d in ds]
    print(f"\n--- [DEBUG] {query}")
    print(f"BM25 : {fmt(bm25_docs)}")
    print(f"向量 : {fmt(vector_docs)}")
    print(f"混合 : {fmt(hybrid_docs)}")
    print(f"精排 : {fmt(final_docs)}")

# ================= 6. 测试 =================
if __name__ == "__main__":
    debug_retrieve("生产环境怎么持久化？")
    ask_enterprise("生产环境怎么持久化？")
    debug_retrieve("Art_1062 的功率是多少？")
    ask_enterprise("Art_1062 的功率是多少？")
