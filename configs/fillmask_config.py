from configparser import ConfigParser

config_parser = ConfigParser()
config_parser.read("./config.ini")
object = config_parser["API_KEY"]
print(object["HF_API"])