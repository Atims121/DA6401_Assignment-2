import torchmetrics
from convnetwork import ConvolutionalNN
import torch
import torchvision as tv
import pytorch_lightning as pl
from operator import attrgetter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dataset = tv.datasets.ImageFolder(
     root='inaturalist_12K/train', transform=tv.transforms.Compose([
    tv.transforms.ToTensor(),
    # tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # tv.transforms.Lambda(lambda x: x.to(device)),
    tv.transforms.Resize((300, 300)),
]),
)
train_data, val_data = torch.utils.data.random_split(dataset, [0.5, 0.5])



# print(tv.models.list_models())
model_name = 'ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1'
model_weights = attrgetter(model_name)(tv.models)
model_architecture = attrgetter(model_name.split('_Weights')[0].lower())(tv.models)

model = model_architecture(weights=model_weights)

model.fc = torch.nn.Linear(model.fc.in_features, 10)
# print('-----------')
# [print(idx, i) for idx, i in enumerate(model.children())]

model_layers = list(model.children())
[layer.requires_grad_(False) for layer in model_layers]

model.fc.requires_grad_(True)

print(model)

optimizer_function = torch.optim.Adam
optimizer_params = {}
accuracy_function = torchmetrics.Accuracy(task="multiclass", num_classes=10)
loss_function = torch.nn.CrossEntropyLoss()

model = ConvolutionalNN(model=model, loss_function=loss_function, accuracy_function=accuracy_function, 
                        optimizer_function=optimizer_function, optimizer_params=optimizer_params)

trainer  = pl.Trainer(log_every_n_steps=5, max_epochs=10)
train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=int(len(train_data)/3))
val_dataloaders = torch.utils.data.DataLoader(val_data)

trainer.fit( model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)




