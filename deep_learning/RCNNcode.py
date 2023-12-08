import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Function to seed the environment for reproducibility
def seed_everything(seed_value):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

seed_everything(42)

# Loading a pre-trained FastRCNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Modifying the classifier to suit the number of classes (2 classes + background)
num_classes = 3
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Training and evaluation setup
from engine import train_one_epoch, evaluate
import utils

# Prepare data loaders
data_loader_train = torch.utils.data.DataLoader(
    # ... dataset and transforms for training ...
)
data_loader_val = torch.utils.data.DataLoader(
    # ... dataset and transforms for validation ...
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Parameters for the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
    # Training for one epoch
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
    lr_scheduler.step()

    # Evaluation on the validation dataset
    evaluate(model, data_loader_val, device=device)

# Saving the trained model
torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')
