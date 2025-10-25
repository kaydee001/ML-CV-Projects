import urllib.request
import tarfile
import os

print("downloading data")
os.makedirs("data", exist_ok=True)

url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
urllib.request.urlretrieve(url, "data/flowers.tgz")

print("extracting data")
with tarfile.open("data/flowers.tgz") as tar:
    tar.extractall("data/")

print("done")