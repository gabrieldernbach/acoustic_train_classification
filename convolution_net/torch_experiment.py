import torch.backends.cudnn

from data_preprocessing import fetch_dummy_dataloader
from models.convolution_net import CNNTiny
from torch_callbacks import BinaryClassificationMetrics, Mixup, SchedulerWrap, SaveCheckpoint
from torch_learner import Learner

# train_dl, validation_dl, test_dl = fetch_balanced_dataloaders(batch_size=250)
train_dl, validation_dl, test_dl = fetch_dummy_dataloader()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

model = CNNTiny()

criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)

callbacks = [BinaryClassificationMetrics(),
             SchedulerWrap(scheduler),
             SaveCheckpoint('ckpt.pt'),
             Mixup(alpha=0.8)]

learner = Learner(model, criterion, optimizer, callbacks=callbacks)
learner.resume('ckpt.pt')
learner.fit(train_dl, validation_dl, max_epoch=200)
