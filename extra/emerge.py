from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSeq2SeqLM
from transformers import T5ForConditionalGeneration, RobertaTokenizer, TrainingArguments, Trainer, TrainerCallback

from peft import PeftModel, PeftConfig

# 加载原始模型
base_model = T5ForConditionalGeneration.from_pretrained("/home/phb/phb/models/codet5-small")

# 加载LoRA模型
peft_model = PeftModel.from_pretrained(base_model, "/home/phb/phb/phb_research/T5/train/results/codet5/checkpoint-49752")
# 合并LoRA权重到基础模型
merged_model = peft_model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("/home/phb/phb/phb_research/T5/emerged_model")

# 如果需要,也可以保存tokenizer
tokenizer = RobertaTokenizer.from_pretrained("/home/phb/phb/models/codet5-small")
tokenizer.save_pretrained("/home/phb/phb/phb_research/T5/emerged_model")