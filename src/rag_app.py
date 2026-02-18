"""
RAG-Enhanced Gradio Demo
Run with: python src/rag_app.py (from project root, requires GPU)
"""
import json
import gradio as gr
from rag_pipeline import RAGPolicyGenerator, detect_services, get_relevant_actions

generator = RAGPolicyGenerator("results/config_a/final_model")

def generate_with_ui(description, max_tokens, temperature, use_rag):
    services = detect_services(description)
    services_str = ", ".join(services) if services else "None detected"
    rag_context = get_relevant_actions(services, description) if use_rag and services else ""
    result = generator.generate(description, max_tokens=int(max_tokens), temperature=temperature, use_rag=use_rag)
    if result["valid_json"]:
        policy_str = json.dumps(result["policy"], indent=2)
        status = "Valid IAM Policy"
    else:
        policy_str = result["raw_output"]
        status = "Invalid JSON - may need manual correction"
    info = f"Services detected: {services_str}\nRAG context: {\'Injected\' if rag_context else \'None\'}"
    return policy_str, status, info

demo = gr.Interface(
    fn=generate_with_ui,
    inputs=[
        gr.Textbox(label="Describe the IAM policy you need", lines=3),
        gr.Slider(minimum=128, maximum=1024, value=512, step=64, label="Max Tokens"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.05, label="Temperature"),
        gr.Checkbox(value=True, label="Enable RAG"),
    ],
    outputs=[
        gr.Code(label="Generated IAM Policy", language="json"),
        gr.Textbox(label="Status"),
        gr.Textbox(label="RAG Info"),
    ],
    title="AWS IAM Policy Generator (RAG-Enhanced)",
    examples=[
        ["Allow read-only access to S3 bucket named customer-data", 512, 0.1, True],
        ["Allow a Lambda function to read from DynamoDB and write logs to CloudWatch", 512, 0.1, True],
        ["Deny all S3 delete operations across all buckets", 512, 0.1, True],
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)
