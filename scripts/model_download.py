from transformers import pipeline

pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
print("Modèle distilbart-cnn-12-6 téléchargé et prêt.")