from transformers import (
    get_scheduler,
)
import torch
import os
import numpy as np
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import Repository, HfApi, HfFolder
import math
from configs.fillmask_config import ConfigHelper, ConfigModel
from mask_model import CustomModel
from data.custom_data import CustomDataset
from preprocessing import Preprocessing

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Used Device: ", device)

class Training():
    def __init__(self, model_name=ConfigModel.MODEL_NAME,
                 learning_rate=ConfigModel.LEARNING_RATE,
                 epoch=ConfigModel.EPOCHS,
                 num_warmup_steps=ConfigModel.NUM_WARMUP_STEPS,
                 name_metric=ConfigModel.METRICs,
                 path_tensorboard=ConfigModel.PATH_TENSORBOARD,
                 path_save=ConfigModel.PATH_SAVE,
                 dataset=None,
                 process=None
                ):
        self.dataset = dataset
        self.process = process
        self.model = CustomModel(model_name).model

        self.epochs = epoch
        self.num_steps = self.epochs * len(self.process.train_loader)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_steps
        )
        #self.metric = evaluate.load(name_metric)
        self.writer = SummaryWriter(path_tensorboard)

        # Define necessary variables
        self.api = HfApi(token=ConfigHelper.TOKEN_HF)
        self.repo_name = path_save  # Replace with your repo name
        self.author = ConfigHelper.AUTHOR
        self.repo_id = self.author + "/" + self.repo_name
        self.token = HfFolder.get_token()
        self.repo = self.setup_hf_repo(self.repo_name, self.repo_id, self.token)

    def setup_hf_repo(self, local_dir, repo_id, token):
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        try:
            self.api.repo_info(repo_id)
            print(f"Repository {repo_id} exists. Cloning...")
        except Exception as e:
            print(f"Repository {repo_id} does not exist. Creating...")
            self.api.create_repo(repo_id=repo_id, token=token, private=True)

        repo = Repository(local_dir=local_dir, clone_from=repo_id)
        return repo

    def save_and_upload(self, epoch, final_commit=False):
        # Save model, tokenizer, and additional files
        self.model.save_pretrained(self.repo_name)
        self.process.tokenizer.save_pretrained(self.repo_name)

        # Push to Hugging Face Hub
        self.repo.git_add(pattern=".")
        commit_message = "Final Commit: Complete fine-tuned model" if final_commit else f"Epoch {epoch}: Update fine-tuned model and metrics"
        self.repo.git_commit(commit_message)
        self.repo.git_push()

        print(f"Model and files pushed to Hugging Face Hub for epoch {epoch}: {self.repo_id}")


    def fit(self, flag_step=False):
        progress_bar = tqdm(range(self.num_steps))
        interval = 200
        for epoch in range(self.epochs):
            # training
            self.model.train()
            n_train_samples = 0
            total_train_loss = 0
            for i, batch in enumerate(self.process.train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                n_train_samples += len(batch)
                outputs = self.model.to(device)(**batch)
                losses = outputs.loss
                losses.backward()

                total_train_loss += round(losses.item(),4)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                if (i + 1) % interval == 0 and flag_step == True:
                    print("Epoch: {}/{}, Iteration: {}/{}, Train Loss: {}".format(
                        epoch + 1,
                        self.epochs,
                        i + 1,
                        len(self.process.train_loader),
                        losses.item())
                    )
                    self.writer.add_scalar('Train/Loss', round(losses.item(),4), epoch * len(self.process.train_loader) + i)

            # evaluate
            self.model.eval()
            n_test_samples = 0
            total_test_loss = 0
            list_losses = []
            for i, batch in enumerate(self.process.test_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                n_test_samples += len(batch)
                with torch.no_grad():
                    outputs = self.model.to(device)(**batch)
                logits = outputs.logits
                losses = outputs.loss
                list_losses.append(losses.repeat(len(batch)))

                total_test_loss += round(losses.item(),4)

    
                if (i + 1) % interval == 0 and flag_step == True:
                    print("Epoch: {}/{}, Iteration: {}/{}, Val Loss: {}".format(
                        epoch + 1,
                        self.epochs,
                        i + 1,
                        len(self.process.train_loader),
                        losses.item())
                    )
                    self.writer.add_scalar('Val/Loss', round(losses.item(),4), epoch * len(self.process.train_loader) + i)
            list_losses = torch.cat(list_losses)
            list_losses = list_losses[: len(self.process.test_loader)]
            try:
                perplexity = math.exp(torch.mean(losses))
            except OverflowError:
                perplexity = float("inf")

            print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

            epoch_train_loss = total_train_loss / n_train_samples
            epoch_test_loss = total_test_loss / n_test_samples
            print(f"train_loss: {epoch_train_loss}  - val_loss: {epoch_test_loss}")

            # Save and upload after each epoch
            final_commit = ((epoch+1) == self.epochs)
            self.save_and_upload((epoch+1), final_commit)

    
if __name__ == '__main__':
    dataset = CustomDataset()
    process = Preprocessing(dataset=dataset.raw_data)
    train = Training(dataset=dataset,process=process)
    train.fit()
