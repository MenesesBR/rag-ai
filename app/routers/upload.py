from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from app.services import pdf_service, keyword_service, mongo_service
from sentence_transformers import SentenceTransformer

router = APIRouter()

embedder = SentenceTransformer('all-MiniLM-L6-v2')

@router.post("/upload/")
async def upload_client_data(
    client_id: str = Form(...),
    persona: str = Form(...),
    files: List[UploadFile] = File(...)
):
    knowledge_chunks = []

    for file in files:
        content = await file.read()
        
        # Extrair texto do PDF
        text = pdf_service.extract_text_from_pdf(content)

        # Dividir o texto em pedaços (chunks)
        chunks = pdf_service.chunk_text(text)

        for chunk in chunks:
            keywords = keyword_service.extract_keywords(chunk)
            embedding = embedder.encode(chunk).tolist()
            knowledge_chunks.append({
                "text": chunk,
                "keywords": keywords,
                "embedding": embedding
            })

    # Salvar client_name e persona
    if not mongo_service.get_client_data(client_id):
        mongo_service.save_client(client_id, persona)

    # Salvar conhecimentos no Mongo
    mongo_service.save_client_knowledge(client_id, knowledge_chunks)

    return {"message": "Dados enviados com sucesso!", "client_id": client_id}
