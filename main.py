import argparse
import os
from urllib.request import urlretrieve
from zipfile import ZipFile

from dataset import dataset
from network import train
#
# batch_size = int
# step = int
# height = int
# width = int

dataset_urls = ["http://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/ut-interaction_segmented_set1.zip",
                "http://cvrc.ece.utexas.edu/SDHA2010/videos/competition_1/ut-interaction_segmented_set2.zip"]

tmp_zip_adr = os.path.join(os.getcwd(), 'dataset/data/tmp.zip')
data_dir = os.path.join(os.getcwd(), 'dataset/data/')


def run():
    if args.update:
        download_dataset_if_needed()
        dataset.create_dataset(data_dir + "/segmented_set?/*.avi")
    print("Begin training")
    train.begin_training()


def download_and_unzip(zipurls):
    for url in zipurls:
        print("Downloading {}".format(url))
        fpath, _ = urlretrieve(url, tmp_zip_adr)

        zf = ZipFile(fpath)
        zf.extractall(data_dir)
        zf.close()
        os.remove(fpath)
    print("Dataset downloaded into 'dataset/data' folder")


def download_dataset_if_needed():
    if os.listdir(data_dir is []):
        print("Data folder is empty. Downloading dataset")
        download_and_unzip(dataset_urls)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-esize", type=int, dest='size', help="Size of examples ")
    parser.add_argument("-estep", type=int, dest='step', help="Size of step for grouping frames into examples")
    parser.add_argument("-height", type=int, dest='height', help="Height of frames")
    parser.add_argument("-width", type=int, dest='width', help="Width of frames")
    parser.add_argument("-u", "--update", action='store_true', help="Re-create tfrecords")

    args = parser.parse_args()
    batch_size = args.size
    step = args.step
    height = args.height
    width = args.width
    run()
