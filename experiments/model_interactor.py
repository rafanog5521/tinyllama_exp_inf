import parameters
from transformers import pipeline


class ModelInteractor:
    def __init__(self):
        self.__pipe__ = pipeline(modality=parameters.modality, model=parameters.model,
                                 torch_dtype=parameters.torch_dtype, device_map=parameters.device_map,
                                 num_return_sequences=parameters.num_return_sequences)

    def prompt(self, question):
        return self.__pipe__.tokenizer.apply_chat_template(question, tokenize=parameters.tokenize,
                                                           add_generation_prompt=parameters.add_generation_prompt)

    def ask_question(self, question):
        prompt = self.prompt(question)
        outputs = self.__pipe__(prompt, max_new_tokens=parameters.max_new_tokens, do_sample=parameters.do_sample,
                                temperature=parameters.temperature, top_k=parameters.top_k, top_p=parameters.top_p)
        answers = ((outputs[0]['generated_text']).split('<|assistant|>\n'))[1]
        return answers
