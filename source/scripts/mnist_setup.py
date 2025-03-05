import os.path
import shutil
import zipfile

from utils.recursive_config import Config


def download_chinese_mnist(config: Config):
    # MNIST dataset
    dataset_download_path = os.path.join(config.get_subpath("data"), "data_template")
    file_name = "chinese-mnist.zip"

    os.makedirs(dataset_download_path, exist_ok=True)

    os.system("kaggle datasets download -d gpreda/chinese-mnist")
    shutil.move(file_name, dataset_download_path)

    zip_path = os.path.join(dataset_download_path, file_name)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_download_path)
    os.remove(zip_path)

    csv_path = os.path.join(dataset_download_path, "chinese_mnist.csv")
    with open(csv_path, "r", encoding="UTF-8") as file:
        lines = file.readlines()
    os.remove(csv_path)

    header = lines[0]
    train_lines = lines[1:12_000]
    val_lines = lines[12_000:]

    train_csv_path = os.path.join(dataset_download_path, "train.csv")
    val_csv_path = os.path.join(dataset_download_path, "val.csv")
    with open(train_csv_path, "wt", encoding="UTF-8") as train, open(val_csv_path, "wt", encoding="UTF-8") as val:
        train.write(header)
        train.writelines(train_lines)
        val.write(header)
        val.writelines(val_lines)


def main():
    os.system("wandb login")
    download_chinese_mnist(Config())


if __name__ == "__main__":
    main()
