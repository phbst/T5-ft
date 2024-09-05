import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,AutoModel,T5ForConditionalGeneration,RobertaTokenizer

def main():
    parser = argparse.ArgumentParser(description="Interact with a merged t5 language model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the merged model")
    args = parser.parse_args()

    # 加载模型和tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)

    # 将模型移动到GPU（如果可用）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Model loaded on {device}. Ready for interaction!")

    # 交互循环
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Goodbye!")
            break

        # 准备输入
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512).to(device)

        # 生成响应
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        # 解码并打印响应
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Model:", response)

if __name__ == "__main__":
    main()