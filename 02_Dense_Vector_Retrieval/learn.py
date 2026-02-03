from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
import asyncio
import nest_asyncio
nest_asyncio.apply()
import os
import openai
from getpass import getpass
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)
from aimakerspace.openai_utils.chatmodel import ChatOpenAI


openai.api_key = getpass("OpenAI API Key: ")
os.environ["OPENAI_API_KEY"] = openai.api_key


text_loader = TextFileLoader("data/HealthWellnessGuide.txt")
documents = text_loader.load_documents()


text_splitter = CharacterTextSplitter()
split_documents = text_splitter.split_texts(documents)

vector_db = VectorDatabase()
vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))

vector_db.search_by_text("What exercises help with lower back pain?", k=3)

chat_openai = ChatOpenAI()
user_prompt_template = "{content}"
user_role_prompt = UserRolePrompt(user_prompt_template)
system_prompt_template = (
    "You are an expert in {expertise}, you always answer in a kind way."
)
system_role_prompt = SystemRolePrompt(system_prompt_template)


RAG_SYSTEM_TEMPLATE = """You are a helpful personal wellness assistant that answers health and wellness questions based strictly on provided context.

    Instructions:
    - Only answer questions using information from the provided context
    - If the context doesn't contain relevant information, respond with "I don't have information about that in my wellness knowledge base"
    - Be accurate and cite specific parts of the context when possible
    - Keep responses {response_style} and {response_length}
    - Only use the provided context. Do not use external knowledge.
    - Include a gentle reminder that users should consult healthcare professionals for medical advice when appropriate
    - Only provide answers when you are confident the context supports your response."""

RAG_USER_TEMPLATE = """Context Information: {context}

Number of relevant sources found: {context_count}
{similarity_scores}

Question: {user_query}

Please provide your answer based solely on the context above."""

rag_system_prompt = SystemRolePrompt(
    RAG_SYSTEM_TEMPLATE,
    strict=True,
    defaults={
        "response_style": "concise",
        "response_length": "brief"
    }
)

rag_user_prompt = UserRolePrompt(
    RAG_USER_TEMPLATE,
    strict=True,
    defaults={
        "context_count": "",
        "similarity_scores": ""
    }
)


class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, vector_db_retriever: VectorDatabase, 
                 response_style: str = "detailed", include_scores: bool = False) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever
        self.response_style = response_style
        self.include_scores = include_scores

    def run_pipeline(self, user_query: str, k: int = 4, **system_kwargs) -> dict:
        # Retrieve relevant contexts
        context_list = self.vector_db_retriever.search_by_text(user_query, k=k)
        
        context_prompt = ""
        similarity_scores = []
        
        for i, (context, score) in enumerate(context_list, 1):
            context_prompt += f"[Source {i}]: {context}\n\n"
            similarity_scores.append(f"Source {i}: {score:.3f}")
        
        # Create system message with parameters
        system_params = {
            "response_style": self.response_style,
            "response_length": system_kwargs.get("response_length", "detailed")
        }
        
        formatted_system_prompt = rag_system_prompt.create_message(**system_params)
        
        user_params = {
            "user_query": user_query,
            "context": context_prompt.strip(),
            "context_count": len(context_list),
            "similarity_scores": f"Relevance scores: {', '.join(similarity_scores)}" if self.include_scores else ""
        }
        
        formatted_user_prompt = rag_user_prompt.create_message(**user_params)

        return {
            "response": self.llm.run([formatted_system_prompt, formatted_user_prompt]), 
            "context": context_list,
            "context_count": len(context_list),
            "similarity_scores": similarity_scores if self.include_scores else None,
            "prompts_used": {
                "system": formatted_system_prompt,
                "user": formatted_user_prompt
            }
        }

rag_pipeline = RetrievalAugmentedQAPipeline(
    vector_db_retriever=vector_db,
    llm=chat_openai,
    response_style="detailed",
    include_scores=True
)

result = rag_pipeline.run_pipeline(
    "What are some natural remedies for improving sleep quality?",
    k=3,
    response_length="comprehensive", 
    include_warnings=True,
    confidence_required=True
)