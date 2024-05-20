import numpy as np
import pandas as pd
import os
import util
import shutil
from datasets import load_dataset, load_metric
from transformers import AutoImageProcessor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer
import torch
from datetime import datetime

CSV_FILE = 'data/raw/airbnb-listings.csv'
IMAGE_FOLDER = 'data/raw/images'
IMAGE_CLASS_FOLDER = 'data/vit_images'

MODEL_CHECKPOINT = "microsoft/beit-large-patch16-224" # pre-trained model from which to fine-tune
BATCH_SIZE = 32 # batch size for training and evaluation


####################
# Data preparation #
####################
df = util.load_data_airbnb(CSV_FILE, pbar_size=494954)

indices_to_remove = []
existing_images = set(os.listdir(IMAGE_FOLDER))

# Remove rows with no corresponding image from df
print("Removing rows without images...")
print("Shape before:", df.shape)
for index, row in df.iterrows():
    image_filename = f"{row['ID']}_thumb.jpg"
    
    if image_filename not in existing_images:
        indices_to_remove.append(index)

df = df.drop(indices_to_remove)
print("Shape after:", df.shape)

# Remove rows which are too recent
print("Removing rows with first review after 2017...")
print("Shape before:", df.shape)
df.dropna(subset=['First Review'], inplace=True)
df.dropna(subset=['Reviews per Month'], inplace=True)
df = df[df['First Review'] <= '2016-12-31']
print("Shape after:", df.shape)


# Assign labels
labels = [0, 1, 2, 3] # Low-high
df['Reviews per Month (Class)'] = pd.qcut(df["Reviews per Month"],
                                           q=len(labels),
                                           labels=labels)
print(df['Reviews per Month (Class)'].value_counts())

grouped = df.groupby('Reviews per Month (Class)')
for label, group in grouped:
    min_value = group['Reviews per Month'].min()
    max_value = group['Reviews per Month'].max()
    print(f"For {label}: Min Value: {min_value}, Max Value: {max_value}")


# Segment images into class folders
shutil.rmtree(IMAGE_CLASS_FOLDER, ignore_errors=True)

os.makedirs(IMAGE_CLASS_FOLDER, exist_ok=True)

for index, row in df.iterrows():
    img_name = f"{row['ID']}_thumb.jpg"
    img_src = os.path.join(IMAGE_FOLDER, img_name)
    label = row['Reviews per Month (Class)']

    target_dir = os.path.join(IMAGE_CLASS_FOLDER, str(label))
    os.makedirs(target_dir, exist_ok=True)

    img_dst = os.path.join(target_dir, img_name)
    shutil.copy2(img_src, img_dst)



dataset = load_dataset("imagefolder", data_dir=IMAGE_CLASS_FOLDER)
image_processor  = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)


normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
if "height" in image_processor.size:
    size = (image_processor.size["height"], image_processor.size["width"])
    crop_size = size
    max_size = None
elif "shortest_edge" in image_processor.size:
    size = image_processor.size["shortest_edge"]
    crop_size = (size, size)
    max_size = image_processor.size.get("longest_edge")

train_transforms = Compose(
        [
            RandomResizedCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(size),
            CenterCrop(crop_size),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(batch):
    """Apply train_transforms across a batch."""
    batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in batch["image"]
    ]
    return batch

def preprocess_test(batch):
    """Apply val_transforms across a batch."""
    batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in batch["image"]]
    return batch

# split up training into training + validation
splits = dataset["train"].train_test_split(test_size=0.1)
train_data = splits['train']
test_data = splits['test']

train_data.set_transform(preprocess_train)
test_data.set_transform(preprocess_test)

# Label converters
labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

model = AutoModelForImageClassification.from_pretrained(
    MODEL_CHECKPOINT,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes = True, # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)

model_name = MODEL_CHECKPOINT.split("/")[-1]

dt_string = datetime.now().strftime("%Y%m%d-%H%M%S")

args = TrainingArguments(
    f"{model_name}-{dt_string}",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

metric = load_metric("accuracy")

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)



def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

trainer = Trainer(
    model,
    args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()
# rest is optional but nice to have
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate()
# some nice to haves:
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
