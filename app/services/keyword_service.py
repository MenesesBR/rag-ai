from keybert import KeyBERT

kw_model = KeyBERT('all-MiniLM-L6-v2')

def extract_keywords(text, top_n=5):
    keywords = kw_model.extract_keywords(text, top_n=top_n)
    return [kw for kw, _ in keywords]
