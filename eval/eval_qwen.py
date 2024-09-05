import json
import torch
import random
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jsonl(file_path, num_samples=1000):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    if len(data) <= num_samples:
        return data
    
    return random.sample(data, num_samples)

def get_model_prediction(model, tokenizer, prompt, device="cuda"):
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    if "true" in response.lower():
        return "true"
    elif "false" in response.lower():
        return "false"
    else:
        return "unknown"

def main():
    model_path = "/home/phb/phb/merge_model/sft-Qwen2-0.5B-Instruct"
    
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    device = "cuda"
    
    logger.info("Loading test data...")
    test_data = load_jsonl("/home/phb/phb/phb_research/data_2/convert/eval_2.jsonl", num_samples=1000)
    
    correct_predictions = 0
    total_predictions = len(test_data)
    unknown_predictions = 0
    
    logger.info(f"Evaluating {total_predictions} samples...")
    for i, item in enumerate(test_data, 1):
        prompt = item["code"]
        expected_output = item["label"].lower()
        
        prediction = get_model_prediction(model, tokenizer, prompt, device)
        
        # 打印每次测试的结果
        logger.info(f"Sample {i}:")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Expected: {expected_output}")
        logger.info(f"Prediction: {prediction}")
        
        if prediction == expected_output:
            correct_predictions += 1
            logger.info("Result: Correct")
        elif prediction == "unknown":
            unknown_predictions += 1
            logger.info("Result: Unknown")
        else:
            logger.info("Result: Incorrect")
        
        logger.info("-----")
        
        # 每10条数据计算一次准确率
        if i % 10 == 0:
            current_accuracy = correct_predictions / i
            logger.info(f"Processed {i}/{total_predictions} samples")
            logger.info(f"Current Accuracy: {current_accuracy:.2%}")
            logger.info("=====")
    
    final_accuracy = correct_predictions / total_predictions
    logger.info(f"Final Accuracy: {final_accuracy:.2%}")
    logger.info(f"Unknown predictions: {unknown_predictions}/{total_predictions} ({unknown_predictions/total_predictions:.2%})")

if __name__ == "__main__":
    main()
