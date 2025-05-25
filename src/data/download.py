import src.common.tools as tools
from urllib.request import urlretrieve
import zipfile


def download_data(link, savepath):
    # Download data from the Web
    urlretrieve(link, savepath)
    # Unzip the data
    with zipfile.ZipFile(savepath, "r") as zip_ref:
        zip_ref.extractall()


if __name__ == "__main__":
    # Download and save the raw data
    config = tools.load_config()
    savename = config["datadirectory"] + config["dataname"] + ".zip"
    download_data(config["datalink"], savename)
