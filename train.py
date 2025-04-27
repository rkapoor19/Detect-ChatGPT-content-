from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load dataset
def load_data():
    with open('data/human_texts.txt', 'r', encoding='utf-8') as f:
        human_texts = f.readlines()
    with open('data/chatgpt_texts.txt', 'r', encoding='utf-8') as f:
        chatgpt_texts = f.readlines()
    texts = human_texts + chatgpt_texts
    labels = [0] * len(human_texts) + [1] * len(chatgpt_texts)
    return texts, labels

# Embed texts
def embed_texts(texts, model):
    return model.encode(texts, convert_to_tensor=False)

def main():
    print("[*] Loading data...")
    texts, labels = load_data()

    print("[*] Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    print("[*] Embedding texts...")
    embeddings = embed_texts(texts, embedder)

    print("[*] Training classifier...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(embeddings, labels)

    print("[*] Saving model...")
    os.makedirs('models/model_checkpoint', exist_ok=True)
    joblib.dump(clf, 'models/model_checkpoint/classifier.joblib')
    embedder.save('models/model_checkpoint/embedder/')

    print("[+] Done.")

if __name__ == "__main__":
    main()
