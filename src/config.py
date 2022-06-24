import torch

DATA_PATH = 'C:\\Users\\Iuliia\\Documents\\Projects\\Contests\\hacs_ai\\izhevsk\\data\\'
IMAGE_SIZE = 256
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RANDOM_SEED = 42
