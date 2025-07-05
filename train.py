from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    Seq2SeqTrainingArguments
)
from datasets import load_dataset

model_name = "./t5-code-explainer"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess(example):
    input_text = f"explain code: {example['sentences']}"
    target_text = example['Explanation']

    input_ids = tokenizer.encode(input_text, truncation=True, max_length=256)
    input_ids = input_ids[:256] + [tokenizer.pad_token_id] * (256 - len(input_ids))
    attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in input_ids]

    labels = tokenizer.encode(target_text, truncation=True, max_length=128)
    labels = labels[:128] + [tokenizer.pad_token_id] * (128 - len(labels))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

dataset = load_dataset("json", data_files={"train": "train-00000-of-00001.json"})
tokenized = dataset.map(preprocess, batched=False)

training_args = Seq2SeqTrainingArguments(
    output_dir="./t5-code-explainer",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    save_total_limit=1,
    predict_with_generate=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("./t5-code-explainer")
tokenizer.save_pretrained("./t5-code-explainer")
