import json
from pydantic import BaseModel
import torch
import transformers
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepeval.models import DeepEvalBaseLLM


class CustomInferenceModel(DeepEvalBaseLLM):
    def __init__(self):
        model_id = 'prometheus-eval/prometheus-8x7b-v2.0'
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            # quantization_config=quantization_config,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id
        )

        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        model = self.load_model()
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_new_tokens=2500,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Create parser required for JSON confinement using lmformatenforcer
        parser = JsonSchemaParser(schema.model_json_schema())
        prefix_function = build_transformers_prefix_allowed_tokens_fn(
            pipeline.tokenizer, parser
        )

        # Output and load valid JSON
        output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
        output = output_dict[0]["generated_text"][len(prompt):]
        json_result = json.loads(output)

        # Return valid JSON object according to the schema DeepEval supplied
        return schema(**json_result)

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Prometheus-eval"
