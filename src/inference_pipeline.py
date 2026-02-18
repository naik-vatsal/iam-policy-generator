"""
IAM Policy Generator - Basic Inference Pipeline (No RAG)
=========================================================
Usage:
    from inference_pipeline import IAMPolicyGenerator
    generator = IAMPolicyGenerator("results/config_a/final_model")
    result = generator.generate("Allow read-only access to S3 bucket my-data")
    print(result["policy"])
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class IAMPolicyGenerator:
    def __init__(self, adapter_path, model_name="mistralai/Mistral-7B-v0.3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map="auto", dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()

    def generate(self, description, max_tokens=512, temperature=0.1):
        prompt = f"### Instruction:\n{description}\n\n### Response:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                do_sample=True, top_p=0.9, repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        policy_text = response.split("### Response:\n")[-1].strip()

        result = {"raw_output": policy_text, "valid_json": False, "policy": None}
        try:
            result["policy"] = json.loads(policy_text)
            result["valid_json"] = True
        except json.JSONDecodeError:
            pass
        return result


if __name__ == "__main__":
    generator = IAMPolicyGenerator("results/config_a/final_model")
    test_cases = [
        "Allow read-only access to S3 bucket named customer-data",
        "Allow a Lambda function to write logs to CloudWatch",
        "Deny all EC2 terminate operations",
    ]
    for desc in test_cases:
        print(f"\nInput: {desc}")
        result = generator.generate(desc)
        if result["valid_json"]:
            print(f"✅ {json.dumps(result['policy'], indent=2)}")
        else:
            print(f"❌ {result['raw_output'][:200]}")