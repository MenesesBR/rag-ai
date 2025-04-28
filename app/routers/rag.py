from fastapi import APIRouter
from typing import List
from app.services import mongo_service
from app.models.schemas import QuestionRequest
from sentence_transformers import SentenceTransformer
import torch
from app.services import openai_service

embedder = SentenceTransformer("all-MiniLM-L6-v2")

router = APIRouter()


@router.post("/ask/")
async def ask_bot(request: QuestionRequest):
    # 0. Verificar cache
    cached_answer = mongo_service.get_answer_cache(request.client_id, request.question)
    if cached_answer:
        return {"answer": cached_answer}

    # 1. Buscar conhecimentos
    knowledge_base = mongo_service.get_client_knowledge(request.client_id)
    if not knowledge_base:
        return {"error": "Cliente não encontrado ou sem conhecimento."}

    persona = mongo_service.get_client_persona(request.client_id)

    # 2. Calcular embedding da pergunta
    question_embedding = embedder.encode(request.question)

    # 3. Buscar top 10 por similaridade
    embeddings = torch.tensor([doc["embedding"] for doc in knowledge_base])
    question_embedding = torch.tensor(question_embedding)
    scores = torch.nn.functional.cosine_similarity(
        embeddings, question_embedding.unsqueeze(0)
    )

    if len(scores) == 0:
        return {"error": "Nenhum conhecimento encontrado para este cliente."}

    k = min(10, len(scores))
    top_indices = scores.topk(k).indices.tolist()

    # 4. Reranking
    candidate_contexts = [knowledge_base[idx]["text"] for idx in top_indices]
    system_reranking_prompt = """
Responda apenas com os números dos melhores trechos separados por vírgula.
Garanta que os números respondidos sejam válidos entre os trechos.
Cada trecho é numerado de 1 a N, onde N é o número total de trechos.
Cada trecho começa com "[INÍCIO DO TRECHO]" e termina com "[FINAL DO TRECHO]".
Exemplo de resposta: 1,4,7.
"""
    reranking_prompt = f"""
Baseado na pergunta: "{request.question}", selecione os 3 trechos mais relevantes abaixo:

{chr(10).join([f"{i+1}. [INÍCIO DO TRECHO {i+1}] {ctx} [FINAL DO TRECHO {i+1}]" for i, ctx in enumerate(candidate_contexts)])}
"""
    rerank_response = openai_service.ask_openai(
        system_reranking_prompt, reranking_prompt
    )
    selected = [
        int(x.strip()) - 1 for x in rerank_response.split(",") if x.strip().isdigit()
    ]

    context = "\n".join([candidate_contexts[i] for i in selected])

    # 5. Criar prompt final
    system_prompt = f"""
Você é um assistente com a seguinte persona: {persona}
Baseie sua resposta apenas nas seguintes informações:

{context}

Caso não tenha certeza ou não tenha informações relevantes, diga que não sabe ou que não tem informações suficientes.
Responda de forma clara e objetiva.
"""
    answer = openai_service.ask_openai(system_prompt, request.question)

    # 6. Salvar no cache
    mongo_service.save_answer_cache(request.client_id, request.question, answer)

    return {"answer": answer}
