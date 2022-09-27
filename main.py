from common.data_loader import DataLoader

DL = DataLoader(1)
data = DL.get_data()

for i in data:
    print(i)
