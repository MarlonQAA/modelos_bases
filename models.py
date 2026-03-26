
import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
  def __init__(self,
               conv_layers = [32,64,128,256],
               lineal_layers = [128],
               n_input_channels = 3,
               n_clases = 10,
               kernel_size = 3
               ):

    super().__init__()

    L = []
    c= n_input_channels

    for n_l in conv_layers:
      L.append(
          nn.Conv2d(
              in_channels=c,
              out_channels=n_l,
              kernel_size=kernel_size,
              stride=1,
              padding=kernel_size
          )
      )

      L.append(nn.BatchNorm2d(num_features=n_l))

      L.append(nn.ReLU())

      L.append(nn.MaxPool2d(kernel_size))

      L.append(nn.Dropout(0.1))

      c = n_l

    self.network = nn.Sequential(*L)


    L2 = []

    for n_l in lineal_layers:
      L2.append(nn.Linear(in_features=c,out_features=n_l) )
      L2.append(nn.ReLU())
      L2.append(nn.Dropout(0.1))
      c=n_l

    L2.append(nn.Linear(in_features=c,out_features=n_clases))
    self.classifier = nn.Sequential(*L2)


  def forward(self,x):
    x = self.network(x)
    x = x.mean(dim = [2,3])
    x = self.classifier(x)
    return x

##### Guarda Pesos del modelo que se empieza a entrenar #####
def save_model(model: nn.Module, name: str):
    from torch import save
    return save(model.state_dict(), name)

#{
#    "conv1.weight": tensor(...),
#    "conv1.bias": tensor(...),
#    "fc.weight": tensor(...),
#    "fc.bias": tensor(...),
#}

##### Carga y mueve el modelo
def load_model(name: str, device=None):
    from torch import load
    #Vemos si existe algun GPU disponible
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    r = CNNClassifier() #Esto construye un modelo nuevo, vacío, con pesos inicializados aleatoriamente.
    r.load_state_dict(load(name, map_location=device, weights_only=True)) # cargas los pesos con load_state_dict() a r.
    r.to(device)
    print(f'Modelo cargado en {device}')
    return r
