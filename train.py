import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import sys

from centerLoss import CenterLoss
from config import *

if torch.cuda.is_available():
    print('GPU is available...')

device = torch.device("cuda")

train_dataset = Data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
train_loader = Data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=2)

# cross entropy loss
criterion = nn.CrossEntropyLoss()
# center loss
criterion_cent = CenterLoss(num_classes=len(classes), feat_dim=feature_dim, use_gpu=torch.cuda.is_available())
# reconstruction loss
criterion_encoder = nn.MSELoss()
# model optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# center loss optimizer
optimizer_cent = optim.Adam(criterion_cent.parameters(), lr=lr)

print(model)

def to_device(device):
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    for state in optimizer_cent.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def train_torch(start_epoch = 1):
    print('Start Training')
    print('Using lam_center {}, lam_encoder {}, feature dimension {}, version {}'.format(lam_center,lam_encoder,feature_dim,version))

    if not os.path.isdir('models/'+version):
        os.mkdir('models/'+version)
    model_batch_path='./models/{}/model_{}d_{}.pkl'
    
    to_device(device)

    model.train()
    for epoch in range(start_epoch,start_epoch+epoch_num):
        model.train()
        for i, data in enumerate(train_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            optimizer_cent.zero_grad()
            outputs = model(inputs)
            labels = torch.max(labels.long(), 1)[1]
            loss_cross = criterion(outputs,labels)
            loss_cent = criterion_cent(model.getSemantic(inputs), labels)
            loss_encoder = criterion_encoder(model.decoder(inputs), inputs)
            loss = loss_cross + lam_center * loss_cent + lam_encoder * loss_encoder
            loss.backward()
            optimizer.step()
            optimizer_cent.step()

            if i == 0:
                print('[%d , %5d] loss: %.3f, center loss: %.3f, reconstruction loss: %.3f' % (epoch + 1, i + 1, loss.item(), loss_cent.item(), loss_encoder.item()))
                
        if epoch >= 50 and epoch % 25 == 0:
            state = { 'model': model.state_dict(), 'cent':criterion_cent.state_dict(), 'optimizer':optimizer.state_dict(),'optimizer_cent':optimizer_cent.state_dict(), 'epoch': start_epoch+epoch_num }
            torch.save(state, model_batch_path.format(version,feature_dim,epoch))

    state = { 'model': model.state_dict(), 'cent':criterion_cent.state_dict(), 'optimizer':optimizer.state_dict(),'optimizer_cent':optimizer_cent.state_dict(), 'epoch': start_epoch+epoch_num }
    torch.save(state, model_path)
    print('Finished Training')

if __name__ == '__main__':
    start_epoch = 1
    if len(sys.argv) > 1 and sys.argv[1] == '-r':
        print('Resuming model from {}'.format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        criterion_cent.load_state_dict(checkpoint['cent'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_cent.load_state_dict(checkpoint['optimizer_cent'])
        start_epoch = checkpoint['epoch']
        
        print('Resume training from epoch {}'.format(start_epoch))

    train_torch(start_epoch)
