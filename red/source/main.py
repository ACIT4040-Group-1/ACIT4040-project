from common.data_loader import DataLoader
from red.source.utils import get_config

if __name__ == '__main__':
    config = get_config()

    DL = DataLoader()
    train_dataset = DL.get_data("train")
    test_dataset = DL.get_data("test")
