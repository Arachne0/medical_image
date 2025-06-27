import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset , DataLoader
from tqdm.auto import tqdm

from make_loader import ImageDataset
from network import Classifier, VGGEncoder
from sklearn.model_selection import train_test_split
from collect_image import ImageDataset , collect_images


class EarlyStopping:
    def __init__(self, patience=3, delta=0.0, verbose=False):
        self.patience = patience  
        self.delta = delta        
        self.verbose = verbose
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_state = {}

    def __call__(self, val_loss, encoder, classifier):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {
                'encoder': encoder.state_dict(),
                'classifier': classifier.state_dict()
            }
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement. Patience: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def main(model_name):
    
    early_stopper = EarlyStopping(patience = 2 , verbose = True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name in ["HAT", "SRFormer"]:
        data_path = f"/home/hail/SH/{model_name}/binary_class"
        label_path = f"/home/hail/SH/{model_name}/labels_train.csv"
    elif model_name == "original":
        data_path = f"/home/hail/SH/medical_image/{model_name}/binary_class"
        label_path = f"/home/hail/SH/medical_image/{model_name}/labels_train.csv"
    else:
        data_path = f"/home/hail/SH/medical_image/{model_name}/binary_class"
        label_path = f"/home/hail/SH/medical_image/{model_name}/labels_train.csv"
    
    collect_image = collect_images(data_root = data_path , label_root = label_path)
    images , labels = collect_image.get_data()

    train_images , test_images , train_labels , test_labels = train_test_split(images , labels , test_size = 0.3 , shuffle = True)
    
    train_test_dict = {
        "train" : [train_images , train_labels],
        "test" : [test_images , test_labels]
    }
    
    for key, value in train_test_dict.items():
        image_dataset = ImageDataset()
        if key == "train":
            train_images , train_labels = image_dataset.get_norm(key , value)
        else: # key == "test"
            test_images , test_labels = image_dataset.get_norm(key , value)
            
    train_dataset = TensorDataset(train_images , train_labels)
    test_dataset = TensorDataset(test_images , test_labels)
    
    train_dataloader = DataLoader(train_dataset , batch_size = 8 , shuffle = True , drop_last= True)
    test_dataloader = DataLoader(test_dataset , batch_size = 8 , shuffle = False , drop_last= True)

    encoder = VGGEncoder()
    encoder.to(device)

    classifier = Classifier()
    classifier.to(device)

    optimizer = optim.Adam([
        {"params" : encoder.parameters() , "lr" : 1e-4},
        {"params" : classifier.parameters() , "lr" : 1e-3}
    ])    
    
    criterion = nn.CrossEntropyLoss()
    
    epochs = 5
    
    logit_ = []
    labels_ = []

    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    for epoch in range(epochs):
        
        encoder.train()
        classifier.train()
        
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images , labels in tqdm(train_dataloader , desc = f"Epoch [{epoch + 1} / {epochs}]"):    
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            z = encoder(images)
            logits = classifier(z)
            loss = criterion(logits , labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits , dim = 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total

        print(f"[Epoch {epoch + 1}] Train_Loss : {train_loss / train_total:.4f} | Train_Acc : {train_acc:.4f}")    
        
        with torch.no_grad():
            encoder.eval()

            test_loss = 0
            test_correct = 0
            test_total = 0
            
            for images , labels in tqdm(test_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                z = encoder(images)
                logits = classifier(z)
                loss = criterion(logits , labels)
                test_loss += loss.item() * labels.size(0)
                preds = torch.argmax(logits , dim = 1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)

                if epoch >= epochs - 10:
                    logit_.append(logits.detach().cpu().numpy())
                    labels_.append(preds.detach().cpu().numpy())
                    
            test_acc = test_correct / test_total
            test_loss /= test_total
            
            print(f"[Epoch {epoch + 1}] Test_loss : {test_loss:.4f} | Test_Acc : {test_acc:.4f}")    
            
        train_loss_list.append(train_loss / train_total)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        early_stopper(test_loss , encoder , classifier)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break

        if model_name in ["HAT", "SRFormer"]:
            torch.save(early_stopper.best_state,f"/home/hail/SH/{model_name}/classify_model.pth")
        elif model_name == "original":
            torch.save(early_stopper.best_state, f"/home/hail/SH/medical_image/{model_name}/classify_model.pth")
        else:
            torch.save(early_stopper.best_state,f"/home/hail/SH/medical_image/{model_name}/classify_model.pth")

        # plt.figure(figsize = (8,6))
        # plt.subplot(1,2,1)
        # plt.plot(train_loss_list , label = "Train_Loss")
        # plt.plot(test_loss_list , label = "Test_Loss")
        # plt.title("Loss over Epoch")
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.grid(True)
        #
        # plt.subplot(1,2,2)
        # plt.plot(train_acc_list , label = "Train_Acc")
        # plt.plot(test_acc_list , label = "Test_Acc")
        # plt.title("Acc over Epoch")
        # plt.xlabel("Epoch")
        # plt.ylabel("Accuracy")
        # plt.legend()
        # plt.grid(True)
        
        # plt.savefig("loss_acc.png")
        
if __name__ == "__main__":
    main("original")