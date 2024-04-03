import torch
from transformers import pipeline

# Crear un pipeline para generación de texto con TinyLLAMA
tinyllama_pipeline = pipeline("text-generation", model="username/tinyllama")

# Entrada del usuario
entrada_usuario = "Hola, ¿cómo estás?"

# Generar respuestas posibles con TinyLLAMA
respuestas = tinyllama_pipeline(entrada_usuario, max_length=50, num_return_sequences=5)

# Imprimir cada respuesta generada
for idx, respuesta in enumerate(respuestas):
    print(f"Respuesta {idx + 1}: {respuesta['generated_text']}")
