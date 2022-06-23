import torch

DATA_PATH = 'data/'
IMAGE_SIZE = 256
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
