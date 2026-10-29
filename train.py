import torch

def train_loop(dataloader,model,optimizer,criterion,device):
  model.train()
  total_corrects=0
  train_loss=0
  total=0

  for x,y in dataloader:
    optimizer.zero_grad()
    x,y=x.to(device),y.to(device)
    Y_hat=model(x)
    loss= criterion(Y_hat,y)
    loss.backward()
    optimizer.step()

    train_loss +=loss.item() * x.size(0)
    total+=y.size(0)
    predicted_classes=Y_hat.argmax(dim=1)
    total_corrects+=(predicted_classes==y).sum().item()

  avg_loss=train_loss/total
  acc=total_corrects/total
  return avg_loss,acc

def train_one_epoch_detection(model,optimizer,dataloader,device,epoch):
  model.train()
  total_loss=0

  for i,(images,targets) in enumerate(dataloader):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict=model(images,targets)
    losses= sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    total_loss+=losses.item()

  avg_loss=total_loss/len(dataloader)
  return avg_loss
if __name__=="__main__":
    pass
