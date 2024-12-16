from transformers import(
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader
from transformers import default_data_collator
from configs.fillmask_config import ConfigModel

class Preprocessing():
    def __init__(self, model_tokenizer=ConfigModel.MODEL_NAME,
                 batch_size=ConfigModel.BATCH_SIZE,
                 chunk_size=ConfigModel.CHUNK_SIZE,
                 train_size=ConfigModel.TRAIN_SIZE,
                 ratio=ConfigModel.RATIO,
                 dataset=None,
                 flag_training=True):
      self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
      self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer,
                                                           mlm_probability=0.15)
      if flag_training:
        print("-"*50, "Information of Tokenizer", "-"*50)
        print(self.tokenizer)
        print("-"*50, "Information of Tokenizer", "-"*50)
        self.chunk_size = chunk_size
        self.tokenized_dataset = self.map_tokenize_dataset(dataset)
        self.group_dataset = self.map_group_text_all_dataset()
        self.downsampled_dataset = self.downsample_dataset(train_size, ratio)
        self.train_loader, self.test_loader = self.data_loader(batch_size)

    def tokenize_dataset(self, sample):
      tokenized_input = self.tokenizer(
          sample["text"]
      )
      if self.tokenizer.is_fast:
        list_word_ids = []
        for i in range(len(tokenized_input["input_ids"])):
          word_id =  tokenized_input.word_ids(i)
          list_word_ids.append(word_id)

        tokenized_input["word_ids"] = list_word_ids
        return tokenized_input

    def map_tokenize_dataset(self, dataset):
      tokenized_dataset = dataset.map(
          self.tokenize_dataset,
          batched=True,
          remove_columns=dataset["train"].column_names
      )
      return tokenized_dataset

    def group_text(self, sample):
      # merge texts
      concat_text = {k: sum(sample[k], []) for k in sample.keys()}

      # total length
      total_length = len(concat_text["input_ids"])

      # remove if < chunk sisze
      total_length = (total_length // self.chunk_size) * self.chunk_size

      # divide by chunk size
      divided_dataset = {
        k: [t[i: i + self.chunk_size] for i in range(0, total_length, self.chunk_size)]
        for k, t in concat_text.items()
      }

      divided_dataset["labels"] = divided_dataset["input_ids"].copy()
      return divided_dataset

    def map_group_text_all_dataset(self):
      group_dataset = self.tokenized_dataset.map(
          self.group_text,
          batched=True
      )
      return group_dataset

    def downsample_dataset(self, train_size, ratio):
      test_size = int(ratio * train_size)
      return self.group_dataset["train"].train_test_split(train_size=train_size, test_size=test_size, seed=42)


    def insert_random_mask(self, sample):
      features = [dict(zip(sample, t)) for t in zip(*sample.values())]
      masked_inputs = self.data_collator(features)
      # Tạo ra một cột "masked" mới cho mỗi cột trong bộ dữ liệu
      return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

    def data_loader(self, batch_size):
      self.downsampled_dataset = self.downsampled_dataset.remove_columns(["word_ids"])
      train_loader = DataLoader(
        self.downsampled_dataset["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=self.data_collator,
      )
      test_loader = self.downsampled_dataset["test"].map(
        self.insert_random_mask,
        batched=True,
        remove_columns=self.downsampled_dataset["test"].column_names,
      )
      test_loader = test_loader.rename_columns(
        {
          "masked_input_ids": "input_ids",
          "masked_attention_mask": "attention_mask",
          "masked_labels": "labels",
        }
      )
      test_loader = DataLoader(
        test_loader, batch_size=batch_size, collate_fn=default_data_collator
      )
      return train_loader, test_loader

