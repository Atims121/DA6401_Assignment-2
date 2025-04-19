import argparse
from operator import attrgetter
import numpy as np
import torchmetrics
from partA.convnetwork import ConvolutionalNN
import torch
import torchvision as tv
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
import os

def main(raw_args=[]):

    print('------- Now begins training -------')
    parser = argparse.ArgumentParser(description='CNN Training Configuration')

    parser.add_argument('-wandb_project', '--wandb_project', type=str, default='myprojectname', help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-wandb_entity', '--wandb_entity', type=str, default='myname', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-wandb_sweepid', '--wandb_sweepid', type=str, default=None, help='Wandb Sweep Id to log in sweep runs the Weights & Biases dashboard.')
    parser.add_argument('-dataset', '--dataset', type=str, default='inaturalist_12K', choices=["inaturalist_12K"], help='Dataset choices: ["inaturalist_12K"]')

    parser.add_argument('-epochs', '--epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=32, help='Batch size for training the model.')
    parser.add_argument('-model_name_weights', '--model_name_weights', type=str, default='ResNet18_Weights.IMAGENET1K_V1', help='Model weights names available in torchvision.models module.')


    # unfreeze first n layers
    parser.add_argument('-unfreeze_first_layers', '--unfreeze_first_layers', type=int, default=1, help='Number of layers to unfreeze from the first layer of the model.')
    # unfreeze last n layers
    parser.add_argument('-unfreeze_last_layers', '--unfreeze_last_layers', type=int, default=1, help='Number of layers to unfreeze from the last layer of the model.')
    parser.add_argument('-num_dense_layers', '--num_dense_layers', type=int, default=1, help='Number of dense layers in the model.') 
    parser.add_argument('-dense_change_ratio', '--dense_change_ratio', type=int, default=4, help='Dense layer size change ratio, if 2 then neurons square root of previous layer.')
    

    args = parser.parse_args(raw_args)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    image_transform = [
        tv.transforms.ToTensor(),
        tv.transforms.Resize((500, 300)),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    
    dataset = tv.datasets.ImageFolder(root='inaturalist_12K/train', transform=tv.transforms.Compose(image_transform))
    train_data, val_data = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    model_name = args.model_name_weights
    model_weights = attrgetter(model_name)(tv.models)
    model_architecture = attrgetter(model_name.split('_Weights')[0].lower())(tv.models)

    model = model_architecture(weights=model_weights)

    model.fc = torch.nn.Linear(model.fc.in_features, 10)

    model_layers = list(model.children())
    # Freezing all the layers of the model for training 
    [layer.requires_grad_(False) for layer in model_layers]
    # Unfreezing the last n layer of the model for training
    
    for layer_index in range(-args.unfreeze_last_layers, 0):
        model_layers[layer_index].requires_grad_(True)
    
    # Unfreezing the first n layer of the model for training
    for layer_index in range(args.unfreeze_first_layers):
        model_layers[layer_index].requires_grad_(True)
        



    optimizer_function = torch.optim.NAdam
    optimizer_params = {}
    accuracy_function = torchmetrics.Accuracy(task="multiclass", num_classes=10)
    loss_function = torch.nn.CrossEntropyLoss()
    model = ConvolutionalNN(model=model, loss_function=loss_function, accuracy_function=accuracy_function, 
                            optimizer_function=optimizer_function, optimizer_params=optimizer_params)
    wandb_logger = WandbLogger(project="CS6910-Assignment-2", reinit=True)
    # log gradients and model topology
    wandb_logger.watch(model)
    trainer  = pl.Trainer(log_every_n_steps=5, max_epochs=args.epochs, logger=wandb_logger)
    train_dataloaders = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)
    val_dataloaders = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size)
    trainer.fit( model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders)


if __name__ == '__main__':
    main()
