import json
from models.pytorch_model import PytorchModelInference
from models.vllm_model import VLLMModelInference
from datasets import load_dataset
from constants import COMPLETENESS_METRIC_TEMPLATE, DEFAULT_EVALUATION_SYSTEM_PROMPT
from helpers import calculate_metrics, parse_score_reason
from openai import OpenAI

# def main():
#     # Load the dataset
#     dataset = load_dataset("GAIR/o1-journey", split='train')
#     print("Dataset loaded with", dataset.num_rows, "entries.")

#     # Initialize the inference model
#     # inference = PytorchModelInference()
#     inference = VLLMModelInference()

#     # Define the system prompt
#     system_prompt = ANSWER_CORRECTNESS_METRIC_TEMPLATE + DEFAULT_PYTORCH_SYSTEM_PROMPT

#     results_dict = {}
#     scores = []

#     # Iterate over each entry in the dataset
#     for idx, entry in enumerate(dataset):
#         input_prompt = entry['question']
#         output_prompt = entry['answer']

#         # Run prediction
#         response = inference.predict(system_prompt, input_prompt, output_prompt)

#         if is_valid_json(response):
#             score, reason = get_data_from_json(response)
#         else:
#             score, reason = -1, 'NULL'

#         # Store results in the results_dict
#         results_dict[idx + 1] = {'score': score, 'reason': reason}
#         scores.append(score)

#     metrics = calculate_metrics(scores)
#     print(metrics)
#     results_dict['metrics'] = metrics

#     # Save the results_dict to a JSON file
#     with open('results_vllm2.json', 'w') as f:
#         json.dump(results_dict, f, indent=4)

#     print("Results saved to 'results_vllm2.json'.")

AUTH_TOKEN = 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJGSjg2R2NGM2pUYk5MT2NvNE52WmtVQ0lVbWZZQ3FvcXRPUWVNZmJoTmxFIn0.eyJleHAiOjE3MzcxOTE3OTMsImlhdCI6MTcwNTY1NTc5MywianRpIjoiMjA4YmUxMTItYmExYS00Zjc0LWFlYjAtNzA3NTE4ODkyZjA1IiwiaXNzIjoiaHR0cDovL2dhdGV3YXkuZTJlbmV0d29ya3MuY29tL2F1dGgvcmVhbG1zL2FwaW1hbiIsImF1ZCI6ImFjY291bnQiLCJzdWIiOiJhY2YzN2ZhOS1jZmIwLTQyNzktYjcwZi03Mjc0MDU0ZjQxZmQiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGltYW51aSIsInNlc3Npb25fc3RhdGUiOiI4NTMwYjNjMy00MWU1LTRhN2EtOGVmZi0wZGQwZDRiZjBhNjQiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImFwaXVzZXIiLCJkZWZhdWx0LXJvbGVzLWFwaW1hbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7ImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInNpZCI6Ijg1MzBiM2MzLTQxZTUtNGE3YS04ZWZmLTBkZDBkNGJmMGE2NCIsImVtYWlsX3ZlcmlmaWVkIjpmYWxzZSwibmFtZSI6IkRldmFuc2ggSmFpbiIsInByaW1hcnlfZW1haWwiOiJkZXZhbnNoLmphaW5AZTJlbmV0d29ya3MuY29tIiwiaXNfcHJpbWFyeV9jb250YWN0Ijp0cnVlLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJkZXZhbnNoLmphaW5AZTJlbmV0d29ya3MuY29tIiwiZ2l2ZW5fbmFtZSI6IkRldmFuc2giLCJmYW1pbHlfbmFtZSI6IkphaW4iLCJlbWFpbCI6ImRldmFuc2guamFpbkBlMmVuZXR3b3Jrcy5jb20ifQ.egSWauw-yC93r1ceZRDEnGy5Xt-Mv-tBaHN5TBl6AJEG94GnuJvtCfWLl6FI9REGdBiIiYkByZcONiSlnporNJru82Z8D5B61uphAeJeS--GUUKNW-hSWHoR43su8hQIqLVdGJ418lF_e-t6ISj3IyLqHpWG_H3SghF6OGtsBVo'


def main():
    # Initialize the OpenAI GenAI client
    client = OpenAI(
        base_url="https://infer.e2enetworks.net/project/p-2164/genai/llama3_2_3b_instruct/v1", 
        api_key=AUTH_TOKEN
    )

    # Define the system prompt
    system_prompt = COMPLETENESS_METRIC_TEMPLATE + DEFAULT_EVALUATION_SYSTEM_PROMPT

    # Load the dataset and print its structure to confirm it has the question and answer fields
    dataset = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split='train')
    # GAIR/o1-journey
    print("Dataset loaded with", dataset.num_rows, "entries.")

    results_dict = {}
    scores = []

    # Iterate over each entry in the dataset
    for idx, entry in enumerate(dataset):
        input_prompt = entry['input']
        output_prompt = entry['output']

        # Prepare the messages to send to GenAI model
        user_input = f"Input: {input_prompt}, Output: {output_prompt}"
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_input},
        ]

        completion = client.chat.completions.create(
            model='llama3_2_3b_instruct',
            messages=messages,
            max_tokens=2048,
        )

        # Extract the model's response from the completion result
        response = completion.choices[0].message.content
        score, reason = parse_score_reason(response)

        # Check if the response is a valid JSON and extract the score and reason
    #     if is_valid_json(response):
    #         score, reason = get_data_from_json(response)
    #     else:
    #         response = sanitize_json_content(response)  # Attempt to sanitize and retry
    #         if is_valid_json(response):  # Check if it became valid JSON
    #             score, reason = get_data_from_json(response)
    #         else:
    #             print(f"JSON INVALID: {response}")
    #             score, reason = -1, 'NULL'

    #     # Store the result in the results_dict
        results_dict[idx + 1] = {'score': score, 'reason': reason}
        if idx % 100 == 0:
            print(idx+1)
        scores.append(score)

    metrics = calculate_metrics(scores)
    print(metrics)
    results_dict['metrics'] = metrics

    # Save the results_dict to a JSON file
    with open('results_genai_4.json', 'w') as f:
        json.dump(results_dict, f, indent=4)

    print("Results saved to results_genai_4.json.")


if __name__ == "__main__":
    main()
