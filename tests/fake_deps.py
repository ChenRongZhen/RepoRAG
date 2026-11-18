"""图与链路的离线路由依赖。"""

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.schemas import IssueHit, IssueSearchResult, RelevanceDecision


def doc(source="article/04-context.md", chunk_id=1, score=0.7, text="上下文压缩分三层。"):
    return Document(
        page_content=text,
        metadata={"source": source, "chunk_id": chunk_id, "score": score, "kind": "doc"},
    )


class FakeRetriever:
    def __init__(self, documents=None, raises=None):
        self.documents = documents if documents is not None else [doc()]
        self.raises = raises
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        if self.raises:
            raise self.raises
        return self.documents


class _Structured:
    def __init__(self, owner):
        self.owner = owner

    def invoke(self, messages):
        self.owner.structured_calls.append(messages)
        if self.owner.structured_raises:
            raise self.owner.structured_raises
        return self.owner.decision


class FakeLLM:
    def __init__(self, decision=None, answer="答案 [D1]", invoke_raises=None,
                 structured_raises=None):
        self.decision = decision or RelevanceDecision(enough_context=True, reason="ok")
        self.answer = answer
        self.invoke_raises = invoke_raises
        self.structured_raises = structured_raises
        self.structured_calls = []
        self.calls = []

    def with_structured_output(self, schema, method=None):
        return _Structured(self)

    def invoke(self, messages):
        self.calls.append(messages)
        if self.invoke_raises:
            raise self.invoke_raises
        return AIMessage(content=self.answer)


class FakeIssueSearch:
    def __init__(self, result=None):
        self.result = result or IssueSearchResult(
            status="ok",
            issues=[IssueHit(number=7, title="崩溃", state="open",
                             url="https://example.com/7", body="正文")],
        )
        self.queries = []

    def __call__(self, question):
        self.queries.append(question)
        return self.result
