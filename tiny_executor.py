import torch
import re
from transformers import pipeline


def verify_cuda():
    if torch.cuda.is_available():
        print("CUDA IS available. Running on GPU")
    else:
        print("CUDA NOT available. Running on CPU")


def __pipe__(num_return_sequences):
    return pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16,
                    device_map="auto", num_return_sequences=num_return_sequences)


def is_question(input_string):
    pattern = r'.*\?$'
    if re.match(pattern, input_string):
        return True
    else:
        return False


def lets_get_the_party_started():
    verify_cuda()
    num_return_sequences = 5  # Cambiar el número según lo necesites
    pipe = __pipe__(num_return_sequences)
    while True:
        q = input("What is your question to TINYLLAMA?[type \"q\" to quit]: ")
        if q.lower() == "q":
            print("Bye bye!")
            break
        if not is_question(q):
            print("\nsorry, I don't recognize that as a question...\n")
        else:
            messages = [
                {
                    "role": "user",
                    "content": q
                }
            ]
            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            for idx, output in enumerate(outputs):
                print(f"Response {idx + 1}: {output['generated_text']}")


if __name__ == "__main__":
    lets_get_the_party_started()
