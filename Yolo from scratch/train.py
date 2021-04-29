import torch
import torchvision.transforms as transforms
import torch.optim as optim 
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolo
from data import VOCDataset
from utils import()


from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

## Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
