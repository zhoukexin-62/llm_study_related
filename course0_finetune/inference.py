from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


final_save_dir = "/data/zhoukexin/Codes/llm_related/final_save_models"

model = AutoModelForCausalLM.from_pretrained(final_save_dir,trust_remote_code=True)
# model.cuda()
tokenizer = AutoTokenizer.from_pretrained(final_save_dir)

# 构建推理流程
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device=6)

prompt = "tell me some singing skill"
generated_texts = pipe(prompt,max_length=512,num_return_sequences=1)

print("\n如是说：-------",generated_texts[0]['generated_text'])