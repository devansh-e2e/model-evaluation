import logging
import json
from datetime import datetime

from e2enetworks.cloud import tir

from constants import EVLUATION_FRAMEWORK_TO_LIST_MAPPING, METRICS_TO_TEMPLATE_MAPPING_DICT, DEFAULT_EVALUATION_SYSTEM_PROMPT, BASE_MODEL_TO_GENAI_MODEL_MAPPING
from helpers import parse_args, get_genai_client, parse_score_reason, calculate_metrics
from dataset_loader import DatasetLoader


logger = logging.getLogger(__name__)


def main(args):
    logger.info("Starting evaluation job with the following script parameters:")
    logger.info(args)

    tir.init()

    # Load the dataset to evaluate
    dataset_loader = DatasetLoader(args)
    dataset = dataset_loader.load()
    logger.info(f"Dataset loaded with {dataset.num_rows} entries.")

    # Get the Genai client to use as a Judge
    client = get_genai_client(args.base_model_id, args.project_id, args.user_auth_token)

    metrics_to_evaluate = EVLUATION_FRAMEWORK_TO_LIST_MAPPING.get(args.model_evaluation_framework)
    for metric_to_evaluate in metrics_to_evaluate:
        results_dict = {}
        scores = []
        metric_template = METRICS_TO_TEMPLATE_MAPPING_DICT.get(metric_to_evaluate)
        # Define the system prompt
        if args.model_evaluation_prompt:
            system_prompt = args.model_evaluation_prompt + metric_template + DEFAULT_EVALUATION_SYSTEM_PROMPT
        else:
            system_prompt = metric_template + DEFAULT_EVALUATION_SYSTEM_PROMPT

        # Iterate over each entry in the dataset
        for idx, entry in enumerate(dataset):
            input_prompt = entry[args.input_column]
            output_prompt = entry[args.output_column]

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
                model=BASE_MODEL_TO_GENAI_MODEL_MAPPING.get(args.base_model_id),
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            # Extract the model's response from the completion result
            response = completion.choices[0].message.content
            score, reason = parse_score_reason(response)

            results_dict[idx + 1] = {'score': score, 'reason': reason}
            scores.append(score)

        metrics = calculate_metrics(scores)
        print(metrics)
        results_dict['metrics'] = metrics

        # Save the results_dict to a JSON file
        RESULTS_JSON = f"results_{metric_to_evaluate}_{datetime.now()}.json"
        with open(RESULTS_JSON, 'w') as f:
            json.dump(results_dict, f, indent=4)

        print(f"Results saved to {RESULTS_JSON}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
