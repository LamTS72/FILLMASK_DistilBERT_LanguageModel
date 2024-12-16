from datasets import load_dataset, concatenate_datasets
from configs.fillmask_config import ConfigDataset
import torch
class CustomDataset():
    def __init__(self,
                 path_dataset=ConfigDataset.PATH_DATASET,
                 revision=ConfigDataset.REVISION,
                 flag_info=True
                ):
        self.raw_data = load_dataset(path_dataset)
        self.size = len(self.raw_data["train"]) + len(self.raw_data["test"]) + len(self.raw_data["unsupervised"])
        if flag_info:
          print("-"*50, "Information of Dataset", "-"*50)
          print(self.raw_data)
          print("-"*50, "Information of Dataset", "-"*50)

    def __len__(self):
      return self.size

    def __getitem__(self, index):
      dataset = concatenate_datasets((self.raw_data["train"],
                          self.raw_data["test"],
                          self.raw_data["unsupervised"]))
      data = dataset[index]["text"]
      target = dataset[index]["label"]
      return {
          "data_text": data,
          "data_label": target
      }

