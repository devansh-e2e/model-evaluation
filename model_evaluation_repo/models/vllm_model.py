from vllm import LLM, SamplingParams
from constants import DEFAULT_VLLM_MAX_TOKENS, DEFAULT_VLLM_MODEL_ID, DEFAULT_VLLM_DEVICE_MAP


class VLLMModelInference():
    def __init__(self, model_id=DEFAULT_VLLM_MODEL_ID, device_map=DEFAULT_VLLM_DEVICE_MAP):
        self.model_id = model_id
        self.device_map = device_map
        self.model, self.sampling_params = self._load_model()

    def _load_model(self):
        sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=DEFAULT_VLLM_MAX_TOKENS)

        model = LLM(model=self.model_id, gpu_memory_utilization=1, max_model_len=4000)
        return model, sampling_params

    def predict(self, system_prompt, input_prompt, output_prompt):
        final_prompt = self._get_prompt(system_prompt, input_prompt, output_prompt)
        response = self.model.chat(final_prompt, self.sampling_params)
        return response[0].outputs[0].text if response else None

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
