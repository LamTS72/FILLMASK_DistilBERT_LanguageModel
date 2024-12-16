from configparser import ConfigParser

config_parser = ConfigParser()
config_parser.read("./config.ini")
object = config_parser["API_KEY"]
# print(object["HF_API"])

class ConfigDataset():
    PATH_DATASET = "imdb"
    REVISION = None


class ConfigModel():
    BATCH_SIZE = 8
    CHUNK_SIZE = 128
    MODEL_NAME = "distilbert-base-uncased"
    TRAIN_SIZE = 10000
    RATIO = 0.1
    LEARNING_RATE = 5e-5
    EPOCHS = 5
    METRICs = "seqeval"
    PATH_TENSORBOARD = "runs/data_run"
    PATH_SAVE = "mask_lm"
    NUM_WARMUP_STEPS = 0

class ConfigHelper():
    TOKEN_HF = object["HF_API"]
    AUTHOR = "Chessmen"