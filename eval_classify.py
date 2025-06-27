import torch
from torch.utils.data import TensorDataset , DataLoader
from tqdm.auto import tqdm

from make_loader import ImageDataset
from collect_image import ImageDataset , collect_images
from network import Classifier, VGGEncoder


def main(model_name):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = VGGEncoder()
    classifier = Classifier()


    if model_name in ["HAT", "SRFormer"]:
        checkpoint = torch.load(f"/home/hail/SH/{model_name}/classify_model.pth")
    elif model_name == "original":
        checkpoint = torch.load(f"/home/hail/SH/medical_image/{model_name}/classify_model.pth")
    else:
        checkpoint = torch.load(f"/home/hail/SH/medical_image/{model_name}/classify_model.pth")

    encoder.load_state_dict(checkpoint["encoder"])
    classifier.load_state_dict(checkpoint["classifier"])
    
    encoder.to(device)
    classifier.to(device)
    
    encoder.eval()
    classifier.eval()

    if model_name in ["HAT", "SRFormer"]:
        data_path = f"/home/hail/SH/{model_name}/binary_class"
        label_path = f"/home/hail/SH/{model_name}/labels_train.csv"
    else:
        data_path = f"/home/hail/SH/medical_image/{model_name}/binary_class"
        label_path = f"/home/hail/SH/medical_image/{model_name}/labels_train.csv"


    collect_image = collect_images(data_root = data_path , label_root = label_path)
    eval_images , eval_labels = collect_image.get_data()
    
    image_dataset = ImageDataset()
    eval_images , eval_labels = image_dataset.get_norm("eval" , [eval_images , eval_labels])
    
    eval_dataset = TensorDataset(eval_images , eval_labels)
    eval_dataloader = DataLoader(eval_dataset , batch_size = 8 , shuffle = False , drop_last= True)    
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images , labels in tqdm(eval_dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            z = encoder(images)
            logits = classifier(z)
            preds = torch.argmax(logits , dim = 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    acc = correct / total
    print(f"Eval Acc {acc :.4f}")    

if __name__ == "__main__":
    main("original")
