from src.dataset import ChestXrayDataset

dataset = ChestXrayDataset("metadata.csv", ".")

print(len(dataset))

img, label = dataset[0]
print(img.shape)
print(label)