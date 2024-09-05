import yaml
import logging
import traceback
import numpy as np
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, RobertaTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class T5Classifier:
    def __init__(self, config_file='../config/config_codet5.yaml'):
        try:
            self.config = self.load_config(config_file)
            logger.info("配置加载成功。")
            self.model = T5ForConditionalGeneration.from_pretrained(self.config['model_path'], device_map="auto")
            self.tokenizer = RobertaTokenizer.from_pretrained(self.config['model_path'])
            logger.info(f"模型和分词器从 {self.config['model_path']} 加载成功")
        except Exception as e:
            logger.error(f"初始化过程中发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def load_config(self, config_file):
        try:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
            
            # 打印所有配置项的类型和值
            for key, value in config.items():
                logger.info(f"{key} - 类型: {type(value)}, 值: {value}")
            
            return config
        except Exception as e:
            logger.error(f"加载配置时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def prepare_datasets(self):
        try:
            logger.info("正在加载数据集...")

            #change
            # dataset = load_dataset("financial_phrasebank", "sentences_allagree")
            dataset = load_dataset("json",data_files=list(self.config['datasets']))

            dataset["train"] = dataset["train"].shuffle(seed=42).select(range(self.config['num_samples']))
            #add
            dataset["train"] = dataset["train"].map(lambda x: {'code': x['code'], 'label': x['label'],'text_label': str(x['label'])})
            #add

            dataset = dataset["train"].train_test_split(test_size=0.005)
            dataset["validation"] = dataset["test"]
            del dataset["test"]
            
            #change
            # classes = dataset["train"].features["label"].names
            # dataset = dataset.map(
            #     lambda x: {"text_label": [classes[label] for label in x["label"]]},
            #     batched=True,
            #     num_proc=1,
            # )





            logger.info("数据集已准备好并分为训练集和验证集。")

            processed_datasets = dataset.map(
                self.preprocess_function,
                batched=True,
                num_proc=1,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=False,
                desc="正在对数据集运行分词器",
            )

            self.train_dataset = processed_datasets["train"]
            self.eval_dataset = processed_datasets["validation"]
            logger.info("数据集已处理并进行分词。")
        except Exception as e:
            logger.error(f"准备数据集时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def configure_lora(self):
        try:
            lora_config = LoraConfig(
                r=int(self.config['lora_r']),
                lora_alpha=int(self.config['lora_alpha']),
                target_modules=["q", "v","k"],
                lora_dropout=float(self.config['lora_dropout']),
                bias="none",
                task_type="SEQ_2_SEQ_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA 配置已应用到模型。")
        except Exception as e:
            logger.error(f"配置 LoRA 时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def train(self):
        try:
            # 检查关键参数的类型和值
            critical_params = ['learning_rate', 'num_train_epochs', 'per_device_train_batch_size', 
                               'per_device_eval_batch_size', 'gradient_accumulation_steps', 
                               'warmup_ratio', 'save_steps', 'logging_steps', 'eval_steps']
            for param in critical_params:
                logger.info(f"{param} - 类型: {type(self.config[param])}, 值: {self.config[param]}")

            training_args = TrainingArguments(
                output_dir=self.config['output_dir'],
                per_device_train_batch_size=int(self.config['per_device_train_batch_size']),
                per_device_eval_batch_size=int(self.config['per_device_eval_batch_size']),
                gradient_accumulation_steps=int(self.config['gradient_accumulation_steps']),
                learning_rate=float(self.config['learning_rate']),
                lr_scheduler_type=self.config['lr_scheduler_type'],
                warmup_ratio=float(self.config['warmup_ratio']),
                num_train_epochs=float(self.config['num_train_epochs']),
                save_strategy=self.config['save_strategy'],
                save_steps=int(self.config['save_steps']),
                logging_steps=int(self.config['logging_steps']),
                evaluation_strategy=self.config['evaluation_strategy'],
                eval_steps=int(self.config['eval_steps'])
            )

            class LoggingCallback(TrainerCallback):
                def on_evaluate(self, args, state, control, metrics, **kwargs):
                    logger.info(f"评估指标: {metrics}")

                def on_log(self, args, state, control, logs=None, **kwargs):
                    if logs:
                        logger.info(f"训练日志: {logs}")
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.compute_metrics,
                callbacks=[LoggingCallback()],
            )
            
            logger.info("开始训练...")
            trainer.train()
            logger.info("训练完成。")

            logger.info("执行最终评估...")
            eval_results = trainer.evaluate()
            logger.info(f"最终评估结果: {eval_results}")

        except Exception as e:
            logger.error(f"训练过程中发生错误: {str(e)}")
            logger.error(f"错误类型: {type(e)}")
            logger.error(f"错误参数: {e.args}")
            logger.error(f"错误追踪:\n{traceback.format_exc()}")
            raise

    def preprocess_function(self, examples):
        try:
            text_column = "code"
            label_column = "text_label"
            inputs = examples[text_column]
            targets = examples[label_column]
            
            # Tokenize inputs with padding and truncation
            model_inputs = self.tokenizer(inputs, max_length=self.config['max_length'], padding="max_length", truncation=True)
            
            # Tokenize targets with padding and truncation
            labels = self.tokenizer(targets, max_length=3, padding="max_length", truncation=True)
            
            # Convert pad_token_id to -100 for labels
            labels_ids = labels["input_ids"]
            labels_ids = [(-100 if token_id == self.tokenizer.pad_token_id else token_id) for token_id in labels_ids]
            
            # Add labels to model_inputs
            model_inputs["labels"] = labels_ids
            
            # Log some sample data
            logger.info(f"Sample input: {inputs[0]}")
            logger.info(f"Sample target: {targets[0]}")
            logger.info(f"Sample model input: {model_inputs['input_ids'][0]}")
            logger.info(f"Sample label: {model_inputs['labels'][0]}")
            
            return model_inputs
        except Exception as e:
            logger.error(f"预处理函数中发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def compute_metrics(self, eval_pred):
        try:
            predictions, labels = eval_pred
            logger.info(f"预测结果类型: {type(predictions)}")
            logger.info(f"标签类型: {type(labels)}")
            
            if isinstance(predictions, tuple):
                # logger.info(f"预测结果是一个包含 {len(predictions)} 个元素的元组")
                # logger.info(f"{predictions}")
                # logger.info(f"{predictions[0]}")
                # logger.info(f"{predictions[0][0]}")
                # 假设第一个元素包含 logits
                predictions = predictions[0]
            
            logger.info(f"预测结果形状: {predictions.shape}")
            logger.info(f"标签形状: {labels.shape}")
            
            # 通过取 argmax 获取预测的类别
            predicted_classes = np.argmax(predictions, axis=-1)
            
            decoded_preds = self.tokenizer.batch_decode(predicted_classes, skip_special_tokens=True)
            # 将 -100 替换为 tokenizer.pad_token_id
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            logger.info(f"样本预测: {decoded_preds[0]}")
            logger.info(f"样本标签: {decoded_labels[0]}")

            # 比较预测和标签
            accuracy = sum([pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels)]) / len(decoded_preds)

            logger.info(f"计算得到的准确率: {accuracy}")

            return {"accuracy": accuracy}
        except Exception as e:
            logger.error(f"compute_metrics 中发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            return {"accuracy": 0.0}  # 返回默认值

def main():
    try:
        classifier = T5Classifier()
        classifier.prepare_datasets()
        classifier.configure_lora()
        classifier.train()
    except Exception as e:
        logger.error(f"main 函数中发生错误: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()