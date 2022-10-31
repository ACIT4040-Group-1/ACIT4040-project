from src.common.DataLoader import DataLoader

DL = DataLoader()
images = DL.get_data()
for i in images:
  print(i[0].numpy())




