from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from typing_extensions import TypedDict


class RAGState(TypedDict):
    question: str
    context: List[Document]
    response: str


# Constants — pricing (USD per million tokens)

# Fireworks AI serverless (https://fireworks.ai/pricing — verify current rates)
FW_CHAT_INPUT_COST_PER_M = 0.22   # gpt-oss-20b input
FW_CHAT_OUTPUT_COST_PER_M = 0.88  # gpt-oss-20b output
FW_EMBED_COST_PER_M = 0.008       # qwen3-embedding-8b

# OpenAI (https://openai.com/api/pricing — verify current rates)
OA_CHAT_INPUT_COST_PER_M = 0.40   # gpt-4.1-mini input
OA_CHAT_OUTPUT_COST_PER_M = 1.60  # gpt-4.1-mini output
OA_EMBED_COST_PER_M = 0.02        # text-embedding-3-small

LANGSMITH_PROJECT = "rag-eval-session16"


# Test set generation from cat-health-guide.pdf
def generate_test_set(test_size: int):
    """Load PDF, chunk it, then use GPT-4.1-mini to generate Q&A pairs per chunk.

    Returns (docs, qa_pairs) where qa_pairs is a list of
    {"question": str, "reference": str} dicts.
    """
    import json
    import random
    import tiktoken
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_openai import ChatOpenAI
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    print(f"[1/5] Generating {test_size} synthetic Q&A pairs from cat-health-guide.pdf...")

    loader = PyMuPDFLoader("data/cat-health-guide.pdf")
    docs = loader.load()
    print(f"    Loaded {len(docs)} pages")

    def _tiktoken_len(text: str) -> int:
        return len(tiktoken.encoding_for_model("gpt-4o").encode(text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=0, length_function=_tiktoken_len
    )
    chunks = splitter.split_documents(docs)
    sampled = random.sample(chunks, min(test_size * 2, len(chunks)))  # over-sample for failures

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    qa_pairs: list[dict] = []
    for chunk in sampled:
        if len(qa_pairs) >= test_size:
            break
        prompt = (
            "You are a test-set generator. Based on the text below, write ONE specific question "
            "that is answerable from the text, and the reference answer.\n\n"
            f"TEXT:\n{chunk.page_content}\n\n"
            'Return ONLY valid JSON with keys "question" and "reference". No markdown, no extra text.'
        )
        try:
            raw = llm.invoke(prompt).content.strip()
            pair = json.loads(raw)
            if "question" in pair and "reference" in pair:
                qa_pairs.append(pair)
        except Exception:
            continue  # skip malformed responses

    print(f"    ✓ Generated {len(qa_pairs)} Q&A pairs")
    for i, qa in enumerate(qa_pairs, 1):
        q = qa["question"]
        print(f"      {i}. {q[:80]}{'...' if len(q) > 80 else ''}")
    print()

    return docs, qa_pairs


#RAG graph builder (shared structure, parameterised by provider)
def build_rag_graph(docs, embedding_model, llm, vector_dim: int, collection_name: str):
    """Return a compiled LangGraph retrieve→generate graph."""
    import tiktoken
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_qdrant import QdrantVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langgraph.graph import START, StateGraph
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    def _tiktoken_len(text: str) -> int:
        return len(tiktoken.encoding_for_model("gpt-4o").encode(text))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=750, chunk_overlap=0, length_function=_tiktoken_len
    )
    chunks = splitter.split_documents(docs)

    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )
    vector_store.add_documents(documents=chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    RAG_PROMPT = (
        "\n#CONTEXT:\n{context}\n\nQUERY:\n{query}\n\n"
        "Use the provide context to answer the provided user query. "
        "Only use the provided context to answer the query. "
        'If you do not know the answer, or it\'s not contained in the provided context respond with "I don\'t know"'
    )
    chat_prompt = ChatPromptTemplate.from_messages([("human", RAG_PROMPT)])
    generator_chain = chat_prompt | llm | StrOutputParser()

    def retrieve(state: RAGState) -> RAGState:
        return {"context": retriever.invoke(state["question"])}

    def generate(state: RAGState) -> RAGState:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        return {"response": generator_chain.invoke({"query": state["question"], "context": docs_content})}

    graph_builder = StateGraph(RAGState).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile(), len(chunks)


#Run pipeline with token + cost tracking
@dataclass
class QueryStats:
    question: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_cost_usd: float = 0.0


@dataclass
class ProviderStats:
    provider: str
    embed_tokens: int = 0
    embed_cost_usd: float = 0.0
    queries: List[QueryStats] = field(default_factory=list)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(q.prompt_tokens for q in self.queries)

    @property
    def total_completion_tokens(self) -> int:
        return sum(q.completion_tokens for q in self.queries)

    @property
    def total_llm_tokens(self) -> int:
        return sum(q.total_tokens for q in self.queries)

    @property
    def total_llm_cost_usd(self) -> float:
        return sum(q.llm_cost_usd for q in self.queries)

    @property
    def grand_total_cost_usd(self) -> float:
        return self.embed_cost_usd + self.total_llm_cost_usd


def run_pipeline(
    graph,
    qa_pairs: list,
    provider_name: str,
    input_cost_per_m: float,
    output_cost_per_m: float,
    langsmith_tag: str,
    query_delay: float = 0.0,
) -> tuple:
    """Run a RAG graph over qa_pairs, tracking tokens per query.

    Returns (EvaluationDataset, ProviderStats).
    """
    import time
    from langchain_community.callbacks import get_openai_callback
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    stats = ProviderStats(provider=provider_name)
    samples: list[SingleTurnSample] = []

    print(f"    Running {provider_name} pipeline over {len(qa_pairs)} questions...")
    for i, qa in enumerate(qa_pairs, 1):
        if query_delay and i > 1:
            time.sleep(query_delay)
        question = qa["question"]
        config = {"metadata": {"provider": langsmith_tag}, "tags": [langsmith_tag]}

        with get_openai_callback() as cb:
            response = graph.invoke({"question": question}, config=config)

        samples.append(SingleTurnSample(
            user_input=question,
            reference=qa["reference"],
            response=response["response"],
            retrieved_contexts=[doc.page_content for doc in response["context"]],
        ))

        llm_cost = (
            cb.prompt_tokens * input_cost_per_m / 1_000_000
            + cb.completion_tokens * output_cost_per_m / 1_000_000
        )
        stats.queries.append(QueryStats(
            question=question,
            prompt_tokens=cb.prompt_tokens,
            completion_tokens=cb.completion_tokens,
            total_tokens=cb.total_tokens,
            llm_cost_usd=llm_cost,
        ))

        print(
            f"      [{i:2d}/{len(qa_pairs)}] "
            f"in={cb.prompt_tokens:>5} out={cb.completion_tokens:>4} "
            f"cost=${llm_cost:.6f}  {question[:55]}..."
        )

    print()
    return EvaluationDataset(samples=samples), stats


def track_embed_tokens(build_fn, embed_cost_per_m: float) -> tuple:
    """Call build_fn() inside get_openai_callback to capture embedding token cost.

    Returns (result, embed_tokens, embed_cost_usd).
    """
    from langchain_community.callbacks import get_openai_callback

    with get_openai_callback() as cb:
        result = build_fn()

    embed_cost = cb.total_tokens * embed_cost_per_m / 1_000_000
    return result, cb.total_tokens, embed_cost


# RAGAS evaluation
def ragas_evaluate(eval_dataset, provider_name: str):
    from langchain_openai import ChatOpenAI
    from ragas import RunConfig, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        LLMContextRecall,
        ResponseRelevancy,
    )

    print(f"    Evaluating {provider_name} with RAGAS...")
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
    result = evaluate(
        dataset=eval_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy()],
        llm=evaluator_llm,
        run_config=RunConfig(timeout=360),
    )
    scores_df = result.to_pandas()
    non_metric = {"user_input", "reference", "response", "retrieved_contexts"}
    avg = {col: scores_df[col].mean() for col in scores_df.columns if col not in non_metric}
    print(f"    ✓ Done — {provider_name}: {avg}\n")
    return avg


#Print results

METRIC_LABELS = {
    "context_recall": "LLM Context Recall",
    "faithfulness": "Faithfulness",
    "factual_correctness(mode=f1)": "Factual Correctness",
    "answer_relevancy": "Response Relevancy",
}


def print_ragas_comparison(fw_result, oa_result) -> None:
    fw = dict(fw_result)
    oa = dict(oa_result)

    header = f"{'Metric':<26} {'Fireworks AI':>14} {'OpenAI 4.1-mini':>16} {'Delta':>8} {'Winner':<10}"
    print(header)
    print("-" * len(header))

    scores_fw, scores_oa = [], []
    for key, label in METRIC_LABELS.items():
        f = fw.get(key, float("nan"))
        o = oa.get(key, float("nan"))
        delta = o - f
        winner = "OpenAI" if delta > 0.005 else ("Fireworks" if delta < -0.005 else "Tie")
        scores_fw.append(f)
        scores_oa.append(o)
        print(f"{label:<26} {f:>14.4f} {o:>16.4f} {delta:>+8.4f}  {winner:<10}")

    avg_fw = sum(scores_fw) / len(scores_fw)
    avg_oa = sum(scores_oa) / len(scores_oa)
    avg_delta = avg_oa - avg_fw
    avg_winner = "OpenAI" if avg_delta > 0.005 else ("Fireworks" if avg_delta < -0.005 else "Tie")
    print("-" * len(header))
    print(f"{'AVERAGE':<26} {avg_fw:>14.4f} {avg_oa:>16.4f} {avg_delta:>+8.4f}  {avg_winner:<10}")


def print_cost_breakdown(fw_stats: ProviderStats, oa_stats: ProviderStats) -> None:
    rows = [
        ("Provider", "Fireworks AI", "OpenAI gpt-4.1-mini"),
        ("Embedding model", "qwen3-embedding-8b", "text-embedding-3-small"),
        ("Chat model", "gpt-oss-20b", "gpt-4.1-mini"),
        ("", "", ""),
        ("Index build", "", ""),
        ("  Embedding tokens", f"{fw_stats.embed_tokens:,}", f"{oa_stats.embed_tokens:,}"),
        ("  Embedding cost", f"${fw_stats.embed_cost_usd:.6f}", f"${oa_stats.embed_cost_usd:.6f}"),
        ("", "", ""),
        ("Inference (all queries)", "", ""),
        ("  Prompt tokens", f"{fw_stats.total_prompt_tokens:,}", f"{oa_stats.total_prompt_tokens:,}"),
        ("  Completion tokens", f"{fw_stats.total_completion_tokens:,}", f"{oa_stats.total_completion_tokens:,}"),
        ("  LLM cost", f"${fw_stats.total_llm_cost_usd:.6f}", f"${oa_stats.total_llm_cost_usd:.6f}"),
        ("", "", ""),
        ("TOTAL COST", f"${fw_stats.grand_total_cost_usd:.6f}", f"${oa_stats.grand_total_cost_usd:.6f}"),
    ]

    col_w = [28, 22, 22]
    header = f"{'Item':<{col_w[0]}} {'Fireworks AI':>{col_w[1]}} {'OpenAI gpt-4.1-mini':>{col_w[2]}}"
    print(header)
    print("-" * sum(col_w))
    for label, fw_val, oa_val in rows:
        if not label and not fw_val and not oa_val:
            print()
            continue
        print(f"{label:<{col_w[0]}} {fw_val:>{col_w[1]}} {oa_val:>{col_w[2]}}")


def print_per_query_cost(fw_stats: ProviderStats, oa_stats: ProviderStats) -> None:
    n = len(fw_stats.queries)
    header = f"  {'#':>2}  {'Fireworks cost':>16}  {'OpenAI cost':>14}  Question"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, (fq, oq) in enumerate(zip(fw_stats.queries, oa_stats.queries), 1):
        print(
            f"  {i:>2}  ${fq.llm_cost_usd:>14.6f}  ${oq.llm_cost_usd:>12.6f}  {fq.question[:55]}..."
        )


def print_langsmith_link() -> None:
    project = os.environ.get("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)
    if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
        print(f"\n🔗 LangSmith traces: https://smith.langchain.com  (project: {project})")
        print(
            "   Filter by tag 'fireworks' or 'openai' to isolate each provider's runs.\n"
            "   The Tokens & Cost tab shows token usage and estimated cost per trace."
        )


def run_comparision() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation: Fireworks AI vs OpenAI")
    parser.add_argument("--test-size", type=int, default=10, help="Number of synthetic test questions (default: 10)")
    args = parser.parse_args()

    load_dotenv()

    # Lazy imports after env is confirmed
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    docs, dataset = generate_test_set(args.test_size)

    print("[2/5] Building Fireworks AI RAG pipeline...")

    fw_embed = OpenAIEmbeddings(
        model=os.environ.get(
            "FIREWORKS_EMBEDDING_MODEL", "accounts/fireworks/models/qwen3-embedding-8b"
        ),
        openai_api_key=os.environ["FIREWORKS_API_KEY"],
        openai_api_base="https://api.fireworks.ai/inference/v1",
        check_embedding_ctx_length=False,
        dimensions=4096,
    )
    fw_llm = ChatOpenAI(
        model=os.environ.get("FIREWORKS_CHAT_MODEL", "accounts/fireworks/models/gpt-oss-20b"),
        openai_api_key=os.environ["FIREWORKS_API_KEY"],
        openai_api_base="https://api.fireworks.ai/inference/v1",
    )

    (fw_graph, fw_chunk_count), fw_embed_tokens, fw_embed_cost = track_embed_tokens(
        lambda: build_rag_graph(docs, fw_embed, fw_llm, 4096, "fireworks_rag"),
        embed_cost_per_m=FW_EMBED_COST_PER_M,
    )
    print(f"Fireworks pipeline ready  ({fw_chunk_count} chunks, "
          f"{fw_embed_tokens:,} embed tokens, cost=${fw_embed_cost:.6f})\n")

 
    print("[3/5] Building OpenAI gpt-4.1-mini RAG pipeline...")

    oa_embed = OpenAIEmbeddings(model="text-embedding-3-small")
    oa_llm = ChatOpenAI(model="gpt-4.1-mini")

    (oa_graph, oa_chunk_count), oa_embed_tokens, oa_embed_cost = track_embed_tokens(
        lambda: build_rag_graph(docs, oa_embed, oa_llm, 1536, "openai_rag"),
        embed_cost_per_m=OA_EMBED_COST_PER_M,
    )
    print(f"OpenAI pipeline ready  ({oa_chunk_count} chunks, "
          f"{oa_embed_tokens:,} embed tokens, cost=${oa_embed_cost:.6f})\n")

    print("[4/5] Running both pipelines over the test set...")

    fw_dataset_copy, fw_stats = run_pipeline(
        fw_graph, dataset,
        provider_name="Fireworks AI",
        input_cost_per_m=FW_CHAT_INPUT_COST_PER_M,
        output_cost_per_m=FW_CHAT_OUTPUT_COST_PER_M,
        langsmith_tag="fireworks",
        query_delay=5,  # avoid Fireworks free-tier rate limit
    )
    fw_stats.embed_tokens = fw_embed_tokens
    fw_stats.embed_cost_usd = fw_embed_cost

    oa_dataset_copy, oa_stats = run_pipeline(
        oa_graph, dataset,
        provider_name="OpenAI gpt-4.1-mini",
        input_cost_per_m=OA_CHAT_INPUT_COST_PER_M,
        output_cost_per_m=OA_CHAT_OUTPUT_COST_PER_M,
        langsmith_tag="openai",
    )
    oa_stats.embed_tokens = oa_embed_tokens
    oa_stats.embed_cost_usd = oa_embed_cost

    print("[5/5] Running RAGAS evaluation...")
    fw_result = ragas_evaluate(fw_dataset_copy, "Fireworks AI")
    oa_result = ragas_evaluate(oa_dataset_copy, "OpenAI gpt-4.1-mini")

    sep = "=" * 75

    print(f"\n{sep}")
    print("RAGAS METRIC COMPARISON")
    print(sep)
    print_ragas_comparison(fw_result, oa_result)

    print(f"\n{sep}")
    print("COST BREAKDOWN")
    print(sep)
    print_cost_breakdown(fw_stats, oa_stats)

    print(f"\n{sep}")
    print("PER-QUERY LLM COST")
    print(sep)
    print_per_query_cost(fw_stats, oa_stats)

    print(f"\n{sep}")
    print("PRICING NOTES")
    print(sep)
    print(f"  Fireworks gpt-oss-20b  : ${FW_CHAT_INPUT_COST_PER_M}/M input, "
          f"${FW_CHAT_OUTPUT_COST_PER_M}/M output")
    print(f"  Fireworks qwen3-emb-8b : ${FW_EMBED_COST_PER_M}/M tokens")
    print(f"  OpenAI gpt-4.1-mini    : ${OA_CHAT_INPUT_COST_PER_M}/M input, "
          f"${OA_CHAT_OUTPUT_COST_PER_M}/M output")
    print(f"  OpenAI text-emb-3-small: ${OA_EMBED_COST_PER_M}/M tokens")
    print(f"  Verify current rates at fireworks.ai/pricing and openai.com/api/pricing")

    print_langsmith_link()
    print()


if __name__ == "__main__":
    run_comparision()
