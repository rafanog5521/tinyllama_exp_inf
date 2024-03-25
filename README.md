# TinyLLama Inference Experiment

Reworded the base chatbot script created [here](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) with some
basic validations to make it a looping mini chatbot.

Includes: 
- Basic regex validation for questions.
- Verbose CUDA validation.
- Quit instruction.

To run it:
- Install nvidia driver.
- Install CUDA 12.3.1v or higher.
- Clone speechgpt repo and install dependencies from "requirements-diego.txt"
- Trigger tiny_executor.py

GPU before running:
![img.png](img.png)

Chatbot running and GPU showing activity:
![img_1.png](img_1.png)