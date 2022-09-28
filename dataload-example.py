from common.data_loader import DataLoader

# example how to use data loader

DL = DataLoader()
data = DL.get_data()

for i in data:
    print(i)
