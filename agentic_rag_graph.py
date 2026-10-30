from typing import TypedDict,Annotated
from operator import add
import re

from langgraph.graph import StateGraph,START,END
from langchain_core.documents import Document

from enterprise_kb_v1 import llm,safe_retrieve,vectorstore
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..",".env"))

# ================= State 定义 =================
MAX_REWRITE=2

class AgenticState(TypedDict,total=False):
    question:str
    current_query:str
    retrieved_docs:list[Document]
    relevance_ok:bool
    rewrite_count:int
    answer:str
    citations:list[dict]

# ================= 节点 1：检索 =================
def retrieve_node(state:AgenticState) -> dict:
    query=state.get("current_query") or state["question"]
    docs=safe_retrieve(query)
    print(f"🔍 检索: query='{query[:50]}' → {len(docs)} 篇")
    return {"retrieved_docs":docs}


# ================= 节点 2：评估充分性 =================
GRADE_PROMPT="""
判断以下检索结果是否包含回答用户问题所需的关键信息。
- 只要结果中有至少 1 条包含能直接回答问题的信息，输出 yes。
- 部分文档不相关不影响判断。
- 只有所有文档都完全无法回答问题时，才输出 no。

问题：{question}
检索结果：
{docs}

只输出 yes 或 no。"""

grade_llm=ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0,
)

def grade_node(state:AgenticState) -> dict:
    # if True:  # 临时：强制不通过，验证循环
    #     print("📊 评估: ❌ 强制不通过（验证循环用）")
    #     return {"relevance_ok": False}
    docs=state.get("retrieved_docs",[])
    if not docs:
        print("📊 评估：无文档 → 不通过")
        return {"relevance_ok":False}

    context="\n".join(
        f"- {d.page_content[:120]}" for d in docs[:5]
    )
    resp=grade_llm.invoke(GRADE_PROMPT.format(
        question=state["question"],docs=context
    ))
    raw=resp.content.strip().lower()
    ok=raw.startswith("yes") or raw.startswith("是")
    print(f"📊 评估: {'✅ 通过' if ok else '❌ 不通过'} (已重写 {state.get('rewrite_count', 0)} 次)")
    return {"relevance_ok":ok}


# ================= 节点 3：重写查询 =================
REWRITE_PROMPT="""用户问题检索效果不佳，请改写查询以提高召回率。

原始问题：{question}
当前查询：{query}
已重写次数：{count}

要求：
1.如果原问题太口语化，转写为更书面、更匹配技术文档风格的表述。
2.如果是多概念问题，拆成更聚焦的单概念查询。
3.只输出改写后的查询，不要解释、不要引号。

改写后的查询："""

def rewrite_node(state:AgenticState) -> dict:
    count=state.get("rewrite_count",0)+1
    current=state.get("current_query") or state["question"]
    resp=llm.invoke(REWRITE_PROMPT.format(
        question=state["question"],query=current,count=count
    ))
    new_query=resp.content.strip().strip('"').strip("'")
    print(f"✏️ 重写 #{count}: {new_query[:60]}")
    return {"current_query":new_query,"rewrite_count":count}


# ================= 路由 =================
def route_after_grade(state:AgenticState) -> str:
    if state.get("relevance_ok"):
        return "generate"
    if state.get("rewrite_count",0)>=MAX_REWRITE:
        print("⚠️ 重写次数耗尽，强制生成（可能触发拒答）")
        return "generate"
    return "rewrite"


# ================= 节点 4：生成 =================
def generate_node(state:AgenticState) -> dict:

    docs=state.get("retrieved_docs",[])
    if not docs:
        return {"answer":"未检索到相关资料。","citations":[]}

    context="\n\n".join(
        f"[{i+1}] {d.page_content}" for i,d in enumerate(docs)
    )
    prompt=f"""严格根据参考资料回答问题。
规则：
1.每个结论后用 [编号] 标注来源，格式如“...用于持久化 [1]。”
2.如果参考资料不足以回答问题，回答“根据现有资料无法确认”。
3.不要编造，不要使用参考资料之外的知识。

参考资料：
{context}

问题：{state['question']}
"""
    answer=llm.invoke(prompt).content

    cited=sorted(set(int(x) for x in re.findall(r"\[(\d+)\]",answer)))
    citations=[]
    for idx in cited:
        if 1<=idx<=len(docs):
            d=docs[idx-1]
            citations.append({
                "index":idx,
                "doc_id":d.metadata.get("doc_id",""),
                "snippet":d.page_content[:80],
            })

    print(f"📝 生成完成，引用 {len(citations)} 条")
    return {"answer":answer,"citations":citations}


# ================= 编译图 =================
def build_graph():
    wf=StateGraph(AgenticState)
    wf.add_node("retrieve",retrieve_node)
    wf.add_node("grade",grade_node)
    wf.add_node("rewrite",rewrite_node)
    wf.add_node("generate",generate_node)

    wf.add_edge(START,"retrieve")
    wf.add_edge("retrieve","grade")
    wf.add_conditional_edges("grade",route_after_grade,{
        "generate":"generate",
        "rewrite":"rewrite",
    })
    wf.add_edge("rewrite","retrieve")
    wf.add_edge("generate",END)

    return wf.compile()


# ================= 入口 =================
if __name__=="__main__":
    graph=build_graph()

    test_questions=[
        "生产环境怎么持久化？",
        "Art_1062 的功率是多少？",
        "LangGraph 的状态持久化用什么组件？",
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"🙋 {q}")
        result=graph.invoke({"question":q})
        print(f"\n🤖 {result['answer']}")
        if result.get("citations"):
            print(f"\n📎 引用:")
            for c in result["citations"]:
                print(f"   [{c['index']}] {c['snippet']}")
