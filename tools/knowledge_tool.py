from langchain_core.tools import tool
from enterprise_kb_v1 import safe_retrieve

@tool
def search_knowledge(query:str) -> str:
    """在企业知识库中检索相关文档。当需要查询公司制度、技术文档、产品参数时使用。"""
    docs=safe_retrieve(query)
    if not docs:
        return "未检索到相关文档。"
    return "\n\n".join(
        f"[{i+1}] {d.page_content}" for i,d in enumerate(docs)
    )