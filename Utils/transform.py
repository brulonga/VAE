from torchvision import transforms
import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset

# Definir la clase de transformación CelebAi
from torchvision import transforms

class CelebAi:
    def __init__(self, image_size=128):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # redimensiona a tamaño fijo
            transforms.ToTensor(),  # convierte a tensor
        ])

    def __call__(self, x):
        return self.transform(x)
    
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob(os.path.join(image_dir, "*"))  # acepta cualquier archivo en la carpeta
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image  # no hay etiquetas

# Asegúrate de que esta clase esté accesible desde el módulo
__all__ = ['CelebAi', 'ImageDataset']

