
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


#### RESNET (desde 0)
import torch.nn.functional as F
class ResNet(nn.Module):
  # Skip-Layers
  #Definimos el Bloque residual
  class Block(nn.Module):
    def __init__(self, n_input, n_output,kernel_size =3,stride =2 ):
      super().__init__() # hereda a nn.Module
      self.c1 = nn.Conv2d(n_input,n_output,
                          kernel_size=kernel_size,
                          padding=kernel_size//2,
                          stride=stride,
                          bias=False)
      self.c2 = nn.Conv2d(n_output,n_output,
                          kernel_size=kernel_size,
                          padding=kernel_size//2,
                          bias=False)
      self.c3 = nn.Conv2d(n_output,n_output,
                          kernel_size=kernel_size,
                          padding=kernel_size//2,
                          bias=False)
      
      self.b1 = nn.BatchNorm2d(n_output) # gamma y betha
      self.b2 = nn.BatchNorm2d(n_output) # gamma y betha
      self.b3 = nn.BatchNorm2d(n_output) # gamma y betha

      self.skip = nn.Conv2d(n_input,n_output,kernel_size=1,stride=stride)

    def forward(self,x):

        #Skip-Layers suminstrar una imagen de alta definicion al F(X)
      return F.relu(
                  self.b3(
                      self.c3(
                          F.relu(
                              self.b2(
                                  self.c2(
                                      F.relu(
                                          self.b1(
                                              self.c1(x)
                                                 )
                                            )
                                          )
                                      )
                                  )
                               ) 
                            )+ self.skip(x)
                      ) 
        
  def __init__(self, layers = [32,64,128,256],linear_layers = [1280], n_input_chanels = 3, n_classes = 10, kernel_size = 3):
    super().__init__()

    L=[]
    c = n_input_chanels
    for l in layers:
      L.append(self.Block(c,l,kernel_size = kernel_size,stride = 2) )
      c = l

    self.network = nn.Sequential(*L)

    L2 = []
    for n_1 in linear_layers:
      L2.append(nn.Linear(c,n_1))
      L2.append(nn.ReLU())
      L2.append(nn.Dropout(p=0.3))
      c = n_1

    L2.append(nn.Linear(c,n_classes))
    self.classifier = nn.Sequential(*L2)

  def forward(self,x):
    x = self.network(x) # [N,256,H,W]
    x = x.mean(dim = [2,3]) # [N,256]
    x = self.classifier(x) # [N,10]
    return(x)


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


##### Carga y mueve el modelo
def load_model_resnet(name: str, device=None):
    from torch import load
    #Vemos si existe algun GPU disponible
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    r = ResNet() #Esto construye un modelo nuevo, vacío, con pesos inicializados aleatoriamente.
    r.load_state_dict(load(name, map_location=device, weights_only=True)) # cargas los pesos con load_state_dict() a r.
    r.to(device)
    print(f'Modelo cargado en {device}')
    return r
