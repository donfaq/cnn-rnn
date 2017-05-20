import tensorflow as tf
import config
import os
from urllib.request import urlretrieve
from zipfile import ZipFile
from dataset.dataset import Dataset
from network.eval import Learning

FLAGS = tf.app.flags.FLAGS
data_dir = config.data_dir
tmp_zip_adr = config.tmp_zip_adr
dataset_urls = config.dataset_urls


def download_dataset_if_needed():
    def download_and_unzip(zipurls):
        for url in zipurls:
            print("Downloading {}".format(url))
            fpath, _ = urlretrieve(url, tmp_zip_adr)
            zf = ZipFile(fpath)
            zf.extractall(data_dir)
            zf.close()
            os.remove(fpath)
        print("Dataset downloaded into 'dataset/data' folder")

    if not os.path.exists(data_dir) or FLAGS.download:
        os.makedirs(data_dir)
        print("Downloading dataset")
        download_and_unzip(dataset_urls)


def main(argv=None):
    download_dataset_if_needed()
    if FLAGS.update or not os.path.exists(data_dir + 'segmented_set1'):
        print("Starting processing binary dataset")
        Dataset().create_dataset(data_dir + "segmented_set?/*.avi")
    Learning()


if __name__ == '__main__':
    tf.app.run()
