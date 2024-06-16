import os
from datasets import load_dataset, load
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from peft import prepare_model_for_int8_training
from peft import LoraConfig, TaskType, get_peft_model
import numpy as np
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
from peft import PeftModel    

HUGGINGFACEAPIKEY="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"

os.environ["WANDB_PROJECT"] = "mlops-proj47"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:512"

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

def train():
    NUM_WORKERS = os.cpu_count()
    NUM_CLASSES = 4
    EPOCHS,R,LORA_ALPHA,LORA_DROPOUT = 1,64,32,0.1
    BATCH_SIZE = 20
    MODEL_ID = "google/gemma-2b"
    dataset = load_dataset('csv', data_files=f'data/processed/traindata.csv')
    dataset['train'] = dataset['train']
    dataset['test'] = dataset['train'].select(range(1000))
    dataset2 = load_dataset('csv', data_files=f'data/processed/testdata2.csv')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenized_dataset = {}

    for split in dataset.keys():
        tokenized_dataset[split] = dataset[split].map(
            lambda x: tokenizer(x["text"], truncation=True), batched=True
        )

    tokenized_dataset2 = {}

    for split in dataset2.keys():
        tokenized_dataset2[split] = dataset2[split].map(
            lambda x: tokenizer(x["text"], truncation=True), batched=True
        )
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID,num_labels=NUM_CLASSES,load_in_8bit=True)
    model = prepare_model_for_int8_training(model)

    lora_model = PeftModel.from_pretrained(model,'fine-tuned-model-full-data-3-epoch',is_trainable=True)
    lora_model.print_trainable_parameters()
    #lora_model = lora_model.merge_and_unload()
    #mark_only_lora_as_trainable(lora_model)
    trainer = Trainer(
        model=lora_model,
        args=TrainingArguments(
            output_dir="./gemmamodel/",
            learning_rate=2e-5,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            evaluation_strategy="epoch",
            save_strategy="no",
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            logging_steps=10,
            report_to="none"
        ),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset2["train"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    print("Evaluating the Model Before Training!")
    print(trainer.evaluate())
    print("Training the Model")
    trainer.train()
    print("Evaluating the Model AFTER Training!")
    #trainer.evaluate()
    print("Saving the model!")
    lora_model.save_pretrained('fine-tuned-model-full-data-4-epoch')







    
    
    
if __name__ == "__main__":
    train()
