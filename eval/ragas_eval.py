"""
v2 端到端评估。运行前确保：
  pip install ragas datasets
"""
import os,sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness,context_precision,answer_relevancy
answer_relevancy.strictness = 1   # 强制只生成 1 个问题，避免 n=3
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..",".env"))
from agentic_rag_graph import build_graph

ragas_llm=ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    n=1,
)
ragas_embeddings=OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
)

def build_eval_dataset(graph):
    cases=[
        {
            "question":"生产环境怎么持久化？",
            "ground_truth":"生产环境推荐使用 SqliteSaver 或 PostgresSaver 进行持久化。",
        },
        {
            "question":"Art_1062 的功率是多少？",
            "ground_truth":"Art_1062 型号的技术参数是 500W 功率。",
        },
        {
            "question":"Python 3.12 有什么新语法？",
            "ground_truth":"Python 3.12 引入了新的类型参数语法。",
        },
    ]

    data={
        "question":[],
        "answer":[],
        "contexts":[],
        "ground_truth":[],
    }

    for c in cases:
        result=graph.invoke({"question":c["question"]})
        data["question"].append(c["question"])
        data["answer"].append(result["answer"])
        data["contexts"].append(
            [d.page_content for d in result.get("retrieved_docs",[])]
        )
        data["ground_truth"].append(c["ground_truth"])

    return Dataset.from_dict(data)

if __name__=="__main__":
    graph=build_graph()
    dataset=build_eval_dataset(graph)

    result=evaluate(
        dataset,
        metrics=[faithfulness,context_precision,answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\n" + "=" * 50)
    print(result)
    print("=" * 50)

    df=result.to_pandas()
    print(df.columns.tolist())
    if "question" not in df.columns:
        df["question"]=["生产环境怎么持久化？", "Art_1062 的功率是多少？", "Python 3.12 有什么新语法？"]
    print(df[["question","faithfulness","context_precision","answer_relevancy"]])
