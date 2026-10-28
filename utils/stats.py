from turtledemo import __main__

from eval import eval_loop
from train import train_loop

def get_model_stats(train_loader,test_loader,eval_loader,model,optimizer,criterion,device,n_epochs,model_name):

  print(f"Inizio addestramento {model_name}")

  for epoch in range(n_epochs):
    train_loss,train_acc=train_loop(train_loader,model,optimizer,criterion,device)
    val_loss,val_acc=eval_loop(eval_loader,model,criterion,device)
    print(
        f"Epoca [{epoch+1:02d}/{n_epochs:02d}] | "
        f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc*100:.2f}% | "
        f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc*100:.2f}%"
    )

  test_loss,test_acc=eval_loop(test_loader,model,criterion,device)
  print("-" * 50)
  print(f"VALUTAZIONE FINALE TEST SET -> Loss: {test_loss:.4f} - Accuracy: {test_acc*100:.2f}%\n")


if __name__=="__main__":
    pass