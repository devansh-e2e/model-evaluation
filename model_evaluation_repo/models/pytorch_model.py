import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from constants import DEFAULT_PYTORCH_MAX_NEW_TOKENS, DEFAULT_PYTORCH_MODEL_ID, DEFAULT_PYTORCH_DEVICE_MAP, DEFAULT_PYTORCH_TASK


class PytorchModelInference():
    def __init__(self, model_id=DEFAULT_PYTORCH_MODEL_ID, quantization_config=None, device_map=DEFAULT_PYTORCH_DEVICE_MAP, task=DEFAULT_PYTORCH_TASK):
        self.model_id = model_id
        self.quantization_config = quantization_config
        self.device_map = device_map
        self.task = task
        self.tokenizer, self.model = self._load_model()

    def _load_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            quantization_config=self.quantization_config,
        )
        return tokenizer, model

    def predict(self, system_prompt, input_prompt, output_prompt):
        pipe = pipeline(
            task=self.task,
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=DEFAULT_PYTORCH_MAX_NEW_TOKENS,
            num_return_sequences=1,
            return_full_text=False,
        )
        final_prompt = self._get_prompt(system_prompt, input_prompt, output_prompt)

        response = pipe(final_prompt)
        return response[0]["generated_text"] if response else None

    def _get_prompt(self, system_prompt, input_prompt, output_prompt):
        user_input = f"Input: {input_prompt}, Output: {output_prompt}"
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": user_input},
        ]
        return messages

    def get_model_name(self):
        return self.model_id
