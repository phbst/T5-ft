import json
import torch
import random
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM,RobertaTokenizer,T5ForConditionalGeneration

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jsonl(file_path, num_samples=1000):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    if len(data) <= num_samples:
        return data
    
    return random.sample(data, num_samples)

def get_model_prediction(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "0" in full_output.lower():
        return "0"
    elif "1" in full_output.lower():
        return "1"
    else:
        return "unknown"

def main():
    model_path = "/home/phb/phb/phb_research/T5/emerged_model"
    
    logger.info("Loading model and tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    logger.info("Loading test data...")
    test_data = load_jsonl("/home/phb/phb/phb_research/data_2/convert/all.jsonl", num_samples=1000)
    
    correct_predictions = 0
    total_predictions = len(test_data)
    unknown_predictions = 0
    
    logger.info(f"Evaluating {total_predictions} samples...")
    for i, item in enumerate(test_data, 1):
        prompt = item["code"]
        expected_output = item["label"].lower()
        
        prediction = get_model_prediction(model, tokenizer, prompt)
        
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