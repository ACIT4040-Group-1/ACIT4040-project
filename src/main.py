from common.DataLoader import DataLoader

if __name__ == '__main__':

    DL = DataLoader()
    train_dataset = DL.get_data("train")
    test_dataset = DL.get_data("test")
    print('hello')