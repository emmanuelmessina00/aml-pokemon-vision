import torch

def eval_loop(dataloader,model,criterion,device):
  model.eval()
  total_corrects=0
  val_loss=0
  total=0

  with torch.no_grad():
    for x,y in dataloader:
      x,y=x.to(device),y.to(device)
      Y_hat=model(x)
      loss= criterion(Y_hat,y)
      val_loss +=loss.item() * x.size(0)
      total+=y.size(0)
      predicted_classes=Y_hat.argmax(dim=1)
      total_corrects+=(predicted_classes==y).sum().item()

  avg_loss=val_loss/total
  acc=total_corrects/total
  return avg_loss,acc