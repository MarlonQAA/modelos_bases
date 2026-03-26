
from sklearn.model_selection import train_test_split

import os
import zipfile
from glob import glob
from pathlib import Path

ZIP_PATH = "/content/modelos_bases/music_instruments.zip"
EXTRACT_PATH = "/content/music_instruments/"

def unzip_dataset(zip_path: str = ZIP_PATH, extract_path: str = EXTRACT_PATH) -> str:
    """
    Descomprime el zip y devuelve la ruta de la carpeta que contiene
    las carpetas de clases.
    """
    extract_path = Path(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    return str(extract_path)

def split_train_val(dataset_origen: str):
  """
  Aqui se crean .csv con las direcciones que apuntan a las
  clases, train, val
  """
  y = [] # nombre de la clase a la que pertence
  X = [] # nombre del archivo de la imagen

  class_names = []

  # nombre de las carpetas
  clases = sorted([
        clase for clase in os.listdir(dataset_origen)
        if os.path.isdir(os.path.join(dataset_origen, clase))
    ])

  for class_id, clase in enumerate(clases):
        class_names.append(clase)
        file_names = glob(os.path.join(dataset_origen, clase, "*.*"))
        for file_name in file_names:
            X.append(file_name) # accordion/0001.jpg
            y.append(class_id) # 0 -> accordion

    # 10 instrumentos * 200 aprox
    # Relativamente Balanceado (% mismo porcentaje de participacion de cada clase)
    # Corte Stratify (asegurarse que haya participacion de cada clase con el porcentaje acordado tanto en
    # el train como el test)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

  with open("/content/class_meaning.csv", 'w', encoding='utf-8') as f:
    for class_id, class_name in enumerate(class_names):
          f.write(f"{class_id};{class_name}\n")

  with open("/content/train_dataset.csv", 'w', encoding='utf-8') as f:
      for (filename, class_) in zip(X_train, y_train):
          f.write(f"{filename};{class_}\n")

  with open("/content/test_dataset.csv", 'w', encoding='utf-8') as f:
      for (filename, class_) in zip(X_test, y_test):
          f.write(f"{filename};{class_}\n")

  print(f"Dataset origen: {dataset_origen}")
  print(f"Total imágenes: {len(X)}")
  print(f"Train: {len(X_train)}")
  print(f"Test: {len(X_test)}")
  print("Archivos generados: class_meaning.csv, train_dataset.csv, test_dataset.csv")


################################################################################

from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from PIL import Image
import torch


class InstrumentsDataset(Dataset):
    def __init__(self, dataset_csv: str, transform=transforms.ToTensor()):
        self.img_names = []
        self.class_idx = []
        self.transform = transform

        with open(dataset_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                img_name, class_id = line.split(";")
                self.img_names.append(img_name)
                self.class_idx.append(int(class_id))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        img_name = self.img_names[idx]
        img = Image.open(img_name).convert("RGB")
        img.load()

        img = self.transform(img)
        class_id = torch.tensor(self.class_idx[idx], dtype=torch.int64)

        return img, class_id

def load_data(dataset_csv: str,
              transform=transforms.ToTensor(),
              num_workers: int = 0,
              batch_size: int = 128,
              shuffle: bool = True):
    dataset = InstrumentsDataset(dataset_csv, transform)

    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )




#if __name__ == "__main__":
#    dataset_path = unzip_dataset(ZIP_PATH, EXTRACT_PATH)
#    split_train_val(dataset_path)
#
#    transform = transforms.Compose([
#    transforms.Resize((140, 140)),
#    transforms.ToTensor()])
#
#    train_loader = load_data("/content/train_dataset.csv",transform = transform, batch_size=32, shuffle=True)
#    test_loader = load_data("/content/test_dataset.csv",transform = transform ,batch_size=32, shuffle=False)
#
#    # Ejemplo de prueba
#    imgs, labels = next(iter(train_loader))
#    print("Batch imágenes:", imgs.shape)
#    print("Batch labels:", labels.shape)


