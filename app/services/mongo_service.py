from pymongo import MongoClient
import os


client = MongoClient(os.getenv("MONGO_URI"))  # ou outro URI do seu Mongo
db = client["rag_ai"]


def save_client_knowledge(client_id, knowledge_chunks):
    for chunk in knowledge_chunks:
        chunk["client_id"] = client_id
    db.knowledge.insert_many(knowledge_chunks)


def get_client_knowledge(client_id):
    docs = list(db.knowledge.find({"client_id": client_id}))
    return docs


def get_client_persona(client_id):
    persona = list(db.client_data.find({"client_id": client_id}))[0]["persona"]
    return persona


def save_client(client_id, persona):
    db.client_data.insert_one({"client_id": client_id, "persona": persona})


def get_client_data(client_id):
    return db.client_data.find({"client_id": client_id})


def save_answer_cache(client_id, question, answer):
    db.answers_cache.insert_one(
        {"client_id": client_id, "question": question, "answer": answer}
    )


def get_answer_cache(client_id, question):
    doc = db.answers_cache.find_one({"client_id": client_id, "question": question})
    return doc["answer"] if doc else None
