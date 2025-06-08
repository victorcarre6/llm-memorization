from transformers import pipeline

pipeline("summarization", model="Falconsai/text_summarization")
print("Modèle text_summarization téléchargé et prêt.")