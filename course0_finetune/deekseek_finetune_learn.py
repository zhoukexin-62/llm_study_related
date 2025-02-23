import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
import os
from data_prepare import samples
from datasets import load_dataset
from transformers import BitsAndBytesConfig 
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments,Trainer

os.environ['CUDA_VISIBLE_DEVICES'] = '6'


# tokenizer op
def tokenize_function(examples):
    # 把sample里面两个拿出来
    texts = [f"{prompt}\n{completion}" for prompt, completion in zip(examples['prompt'], examples['completion'])]
    # 进行tokenizer
    token = tokenizer(texts, padding="max_length", truncation=True, max_length=1024) # 截断、最大长度
    # labels/target---input_ids
    token["labels"] = token["input_ids"].copy()

    return token


# 1.load

model_name = "/data/zhoukexin/Models/Deepseek-R1-1.5B" # 找一个小一点的
model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)
model.cuda()

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("model loaded-----")

# 2.data json prepare
with open("datasets.jsonl","w",encoding="utf-8") as f:
    for s in samples:
        json_line =json.dumps(s,ensure_ascii=False)
        f.write(json_line + "\n")
    else:
        print("prepare data finished-----")

# 3.data split
ds = load_dataset("json", data_files={"train":"datasets.jsonl"},split='train') # 配准train进行split
print("dataset size : ",len(ds)) # 50 case

train_test_split = ds.train_test_split(test_size=0.1)
train_ds = train_test_split['train']
test_ds = train_test_split['test']

print("train dataset size : ",len(train_ds))
print("test dataset size : ",len(test_ds))

print("data split finished-------")

# 4.tokenize--map
tokenized_train_ds = train_ds.map(tokenize_function, batched=True)  
tokenized_test_ds = test_ds.map(tokenize_function, batched=True)

print('tokenize done--------')
print(tokenized_train_ds[0])

# prompt
# completion
# input_ids
# attention_mask
# labels

# 5.量化设置

quantization_config = BitsAndBytesConfig(load_in_4bit = True) # 4B/8B

model = AutoModelForCausalLM.from_pretrained(model_name,quantization_config=quantization_config,device_map= "auto")

print('quantization done--------')


# 6. lora 微调

lora_config = LoraConfig(
    r=8,  
    lora_alpha=16,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, 
    task_type=TaskType.CAUSAL_LM) # lora配置

model = get_peft_model(model, lora_config) # lora微调model
model.print_trainable_parameters() # 打印可训练参数
print('lora done--------')


# 7. 训练参数设置

train_agrs = TrainingArguments(
    output_dir = "./finetune_models",#
    overwrite_output_dir = True,
    num_train_epochs = 10,#
    per_device_train_batch_size = 2, #
    per_device_eval_batch_size = 2,
    evaluation_strategy = "steps",
    save_steps = 100, #
    eval_steps = 10, #
    logging_steps = 1000,
    learning_rate = 3e-5,#
    weight_decay = 0.01,
    warmup_steps = 1000,
    logging_dir = "./logs",#
    report_to = "none",
    gradient_accumulation_steps= 8,#
    fp16=True, #
    run_name= 'deepseek-r1-1.5b-finetune' #

)

print('train args done--------')

# 8. 训练

trainer = Trainer(
    model = model, # lora ft model
    args = train_agrs,
    train_dataset = tokenized_train_ds,
    eval_dataset = tokenized_test_ds
) # 定义训练器


print('trainer start--------')
trainer.train() # 开始训练

# 9. 保存模型

# lora 模型保存
save_dir = "./save_models"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print('lora model saved--------')

# 全量模型保存

final_save_dir = "./final_save_models"
base_model = AutoModelForCausalLM.from_pretrained(model_name) # 基础模型

model = PeftModel.from_pretrained(base_model, save_dir)
model = model.merge_and_unload() # 合并模型

model.save_pretrained(final_save_dir)
tokenizer.save_pretrained(final_save_dir)

print('final model saved--------')
