import torch
from torchvision import transforms
from torchvision.datasets import MNIST, USPS, SVHN
from torch.utils.data import DataLoader, ConcatDataset,random_split
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from networks import LeNet, ResNet20, DenseNet40
from metrics import equal_interval_ece
from losses import ECELoss,ECLoss_hd
from tqdm import tqdm

from Utils import get_weight_union,TransCal_Fun
from Utils import drl_boost
from Utils import pseudocal

from scipy.optimize import fmin
from scipy.optimize import minimize

DEVICE = "cuda:0"

def Load_Data(target_domain="MNIST"):

    transform_gray2rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  
        transforms.Resize((28, 28))
    ])
    transform_svhn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
    ])
    
    mnist_train = MNIST(root='./data/MNIST', train=True, transform=transform_gray2rgb, download=True)
    mnist_test  = MNIST(root='./data/MNIST', train=False, transform=transform_gray2rgb, download=True)
    mnist_domain = ConcatDataset([mnist_train, mnist_test])

    usps_train = USPS(root='./data/USPS', train=True, transform=transform_gray2rgb, download=True)
    usps_test  = USPS(root='./data/USPS', train=False, transform=transform_gray2rgb, download=True)
    usps_domain = ConcatDataset([usps_train, usps_test])

    svhn_train = SVHN(root='./data/SVHN', split='train', transform=transform_svhn, download=True)
    svhn_test  = SVHN(root='./data/SVHN', split='test', transform=transform_svhn, download=True)
    svhn_domain = ConcatDataset([svhn_train, svhn_test])

    if target_domain == "MNIST":
        Source_domain = ConcatDataset([usps_domain,svhn_domain])
        Target_domain = mnist_domain
    elif target_domain == "USPS":
        Source_domain = ConcatDataset([mnist_domain,svhn_domain])
        Target_domain = usps_domain
    elif target_domain == "SVHN":
        Source_domain = ConcatDataset([mnist_domain,usps_domain])
        Target_domain = svhn_domain

    batch_size = 256
    Source_loader = DataLoader(Source_domain, batch_size=batch_size, shuffle=True,num_workers=4)
    Target_loader  = DataLoader(Target_domain, batch_size=batch_size, shuffle=True,num_workers=4)

    return Source_domain,Target_domain,Source_loader,Target_loader


if __name__=="__main__":

    target_domain = "MNIST"   #MNIST,USPS,SVHN
    weight = r""
    run_Uncal_ECE = True
    run_ECE_TS = True
    run_TransCal = True
    run_DRL = True
    run_PseudoCali = True
    run_ECL_TS = True
    run_Oracle_TS = True

    # Load Data
    Source_domain,target_domain_dataset,Source_loader,Target_loader = Load_Data(target_domain=target_domain)
    
    # Load Model
    # model = LeNet().to(DEVICE)
    model = ResNet20().to(DEVICE)
    # model = DenseNet40().to(DEVICE)

    print("Experiment on "+model.name+" on "+target_domain)

    if weight == "":
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        optimizer2 = torch.optim.Adam(model.classifier2.parameters(), lr=0.01)
        num_epochs = 100
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_loss2 = 0.0
            for iter,(batch_X, batch_y) in enumerate(Source_loader):
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, dim=1).detach()
                index = preds==batch_y
                weight = torch.tensor([1-sum(index)/(len(index)+1e-8),sum(index)/(len(index)+1e-8)]).to(preds.device)
                optimizer2.zero_grad()
                outputs2 = model.forward_classifier2(batch_X)
                loss2 = F.cross_entropy(outputs2, (preds==batch_y).long(), weight=weight)
                loss2.backward()
                optimizer2.step()

                epoch_loss += loss.item()
                epoch_loss2 += loss2.item()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(Source_loader):.4f}, Loss2: {epoch_loss2 / len(Source_loader):.4f}")
        torch.save(model.state_dict(), "weights/Digit/"+model.name+"_"+target_domain+".pth")
    else:
        model.load_state_dict(torch.load(weight,weights_only=True))
        model = model.to(DEVICE)

    if run_Uncal_ECE:
        model.eval()
        with torch.no_grad():
            test_preds_list = []
            test_labels_list = []
            test_probs_list = []
            for batch_X, batch_y in Target_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                test_logits = model(batch_X) 
                test_probs_all = torch.softmax(test_logits, dim=1)
                test_preds = torch.argmax(test_probs_all, dim=1)
                test_probs = test_probs_all.max(dim=1).values
                test_preds_list.append(test_preds)
                test_labels_list.append(batch_y)
                test_probs_list.append(test_probs)
        
        test_preds = torch.cat(test_preds_list,dim=0).cpu().numpy()
        test_labels = torch.cat(test_labels_list,dim=0).cpu().numpy()
        test_probs = torch.cat(test_probs_list,dim=0).cpu().numpy()
        test_accuracy = np.mean(test_preds == test_labels)
        print("Test Accuracy: {:.4f}".format(test_accuracy))

        # compute Uncal ECE
        ece, _, _, _, _ = equal_interval_ece(test_probs, test_preds==test_labels, num_bins=15)
        print(f"Uncal ECE: {ece:.4f}")

    if run_ECE_TS:
        train_logit_list = []
        Train_label_list = []
        model.eval()
        with torch.no_grad():
            for batch_X, batch_y in Source_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                train_logits = model(batch_X) 
                train_logit_list.append(train_logits)
                Train_label_list.append(batch_y)
        train_logits = torch.cat(train_logit_list,dim=0)
        train_labels = torch.cat(Train_label_list,dim=0)

        def calibration_loss(T, train_logits, Train_labels):
            train_logits = train_logits/T.item()

            train_preds = torch.argmax(train_logits, dim=1)
            train_probs = torch.softmax(train_logits, dim=1).max(dim=1).values
            Train_H = train_preds==Train_labels

            ece = ECELoss()(train_probs, Train_H)
            #print(f"T: {T.item():4f}")

            return ece

        def objective_temperature(T):
            return calibration_loss(T, train_logits, train_labels).item()
        
        initial_temperature = [10.0]
        bounds = [(1.0,10.0)]
        best_temperature = minimize(objective_temperature, initial_temperature, method='Powell',bounds=bounds,options={'disp': True}).x[0].item()

        print(f"best_temperature: {best_temperature:4f}")

        with torch.no_grad():
            test_preds_list = []
            test_labels_list = []
            test_probs_list = []
            for batch_X, batch_y in Target_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                test_logits = model(batch_X) 
                test_logits = test_logits/best_temperature
                test_probs_all = torch.softmax(test_logits, dim=1)
                test_preds = torch.argmax(test_probs_all, dim=1)
                test_probs = test_probs_all.max(dim=1).values
                test_preds_list.append(test_preds)
                test_labels_list.append(batch_y)
                test_probs_list.append(test_probs)

        # compute ECE_TS's ECE
        test_preds = torch.cat(test_preds_list,dim=0).cpu().numpy()
        test_labels = torch.cat(test_labels_list,dim=0).cpu().numpy()
        test_probs = torch.cat(test_probs_list,dim=0).cpu().numpy()
        ece, _, _, _, _ = equal_interval_ece(test_probs, test_preds==test_labels, num_bins=15)
        index = test_preds==test_labels
        acc = sum(index)/len(index)
        print(f"ECE_TS's ECE: {ece:.4f}")

    if run_TransCal:
        train_size = int(0.8 * len(Source_domain))
        val_size = len(Source_domain) - train_size

        # 使用 random_split 进行数据集划分
        train_dataset, val_dataset = random_split(Source_domain, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # get features
        model.eval()
        with torch.no_grad():
            train_features = []
            train_y = []
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                train_feature = model.get_features(batch_X) 
                train_features.append(train_feature)
                train_y.append(batch_y)
            train_features = torch.cat(train_features,dim=0)
            train_features = train_features.view(train_features.size(0), -1).cpu().numpy()
            train_y = torch.cat(train_y,dim=0)
            val_features = []
            val_y = []
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                test_feature = model.get_features(batch_X) 
                val_features.append(test_feature)
                val_y.append(batch_y)
            val_features_tensor = torch.cat(val_features,dim=0)
            val_features = val_features_tensor.view(val_features_tensor.size(0), -1).cpu().numpy()
            val_y = torch.cat(val_y,dim=0).cpu()

            test_features = []
            test_y = []
            for batch_X, batch_y in Target_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                test_feature = model.get_features(batch_X) 
                test_features.append(test_feature)
                test_y.append(batch_y)
            test_features_tensor = torch.cat(test_features,dim=0)
            test_features = test_features_tensor.view(test_features_tensor.size(0), -1).cpu().numpy()
            test_y = torch.cat(test_y,dim=0).cpu().numpy()
        
            weight = get_weight_union(train_features,test_features,val_features)

            logits_source_val = model.classify(val_features_tensor).cpu()
            logits_target = model.classify(test_features_tensor).cpu()
            optimal_temp = TransCal_Fun(logits_source_val, val_y, logits_target, weight)

            logits_target = logits_target/optimal_temp
            test_probs_all = torch.softmax(logits_target, dim=1)
            test_preds = torch.argmax(test_probs_all, dim=1).cpu().numpy()
            test_probs = test_probs_all.max(dim=1).values.cpu().numpy()
            ece, _, _, _, _ = equal_interval_ece(test_probs, test_preds==test_y, num_bins=15)
            print(f"TranCal's ECE: {ece:.4f}")
            print("!")

    if run_DRL:
        train_size = int(0.8 * len(Source_domain))
        val_size = len(Source_domain) - train_size

        # 使用 random_split 进行数据集划分
        train_dataset, val_dataset = random_split(Source_domain, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True,drop_last=True)
        model_alpha,model_beta = drl_boost(train_loader,val_loader,model,DEVICE)

        target_loader  = DataLoader(target_domain_dataset, batch_size=256, shuffle=True,num_workers=4,drop_last=True)
        with torch.no_grad():
            test_preds_list = []
            test_labels_list = []
            test_probs_list = []
            for batch_X, batch_y in target_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                pred = F.softmax(model_beta(batch_X, None, None, None, None).detach(), dim=1)
                r_target = (pred[:, 0] / pred[:, 1]).reshape(-1, 1)
                test_logits = model_alpha(batch_X, torch.ones((256, 10)).to(DEVICE), r_target.to(DEVICE)).detach()
                test_probs_all = torch.softmax(test_logits, dim=1)
                test_preds = torch.argmax(test_probs_all, dim=1)
                test_probs = test_probs_all.max(dim=1).values
                test_preds_list.append(test_preds)
                test_labels_list.append(batch_y)
                test_probs_list.append(test_probs)

        # compute DRL's ECE
        test_preds = torch.cat(test_preds_list,dim=0).cpu().numpy()
        test_labels = torch.cat(test_labels_list,dim=0).cpu().numpy()
        test_probs = torch.cat(test_probs_list,dim=0).cpu().numpy()
        ece, _, _, _, _ = equal_interval_ece(test_probs, test_preds==test_labels, num_bins=15)
        print(f"DRL's ECE: {ece:.4f}")

    if run_PseudoCali:
        soft_t, hard_t = pseudocal(Target_loader,model,DEVICE,10)

        model.eval()
        with torch.no_grad():
            test_preds_list = []
            test_labels_list = []
            test_probs_list = []
            for batch_X, batch_y in Target_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                test_logits = model(batch_X) 
                test_logits = test_logits/soft_t
                test_probs_all = torch.softmax(test_logits, dim=1)
                test_preds = torch.argmax(test_probs_all, dim=1)
                test_probs = test_probs_all.max(dim=1).values
                test_preds_list.append(test_preds)
                test_labels_list.append(batch_y)
                test_probs_list.append(test_probs)
        
        test_preds = torch.cat(test_preds_list,dim=0).cpu().numpy()
        test_labels = torch.cat(test_labels_list,dim=0).cpu().numpy()
        test_probs = torch.cat(test_probs_list,dim=0).cpu().numpy()

        # compute PseudoCali's ECE
        ece, _, _, _, _ = equal_interval_ece(test_probs, test_preds==test_labels, num_bins=15)
        print(f"PseudoCali's ECE: {ece:.4f}, soft_t: {soft_t:.4f}")

    if run_ECL_TS:
        
        train_X_list = []
        test_X_list = []
        train_logit_list = []
        test_logit_list = []
        Train_label_list = []

        model.eval()
        with torch.no_grad():
            for (train_X, train_y), (test_X, _) in tqdm(zip(Source_loader, Target_loader),total=min(len(Source_loader),len(Target_loader))):
                train_X = train_X.to(DEVICE)
                test_X = test_X.to(DEVICE)
                train_y = train_y.to(DEVICE)
                train_output = model(train_X)
                test_output = model(test_X)

                train_X_list.append(train_X)
                test_X_list.append(test_X)
                train_logit_list.append(train_output)
                test_logit_list.append(test_output)
                Train_label_list.append(train_y)

        train_X = torch.cat(train_X_list,dim=0)
        test_X = torch.cat(test_X_list,dim=0)
        train_logits = torch.cat(train_logit_list,dim=0)
        test_logits = torch.cat(test_logit_list,dim=0)
        Train_labels = torch.cat(Train_label_list,dim=0)

        la_ece = 0.
        la_Lce = 0.

        def calibration_loss(T, train_logits, Train_labels, model,train_X,test_X,test_logits):
            train_logits = train_logits/T
            test_logits = test_logits/T

            train_preds = torch.argmax(train_logits, dim=1)
            train_probs = torch.softmax(train_logits, dim=1).max(dim=1).values
            test_probs = torch.softmax(test_logits, dim=1).max(dim=1).values
            Train_H = train_preds==Train_labels

            ece = ECELoss()(train_probs, Train_H)
            ECLoss = ECLoss_hd()
            L_ce = ECLoss(model,train_X,test_X,train_probs,test_probs)
            global la_ece
            global la_Lce

            la_ece = la_ece + ece
            la_Lce = la_Lce + L_ce
            la = la_ece/la_Lce
            value = ece+la*(1/(ece**2+1e-8))*L_ce
            print(f"T: {T:.4f}, ece: {ece:.4f}, Lce: {L_ce:4f}, la: {la:.4f}, loss: {value:.4f}")

            return value

        def objective_temperature(T):
            return calibration_loss(T, train_logits, Train_labels, model,train_X,test_X,test_logits).item()

        Ts = [1.0 +i for i in range(50)]
        best_temperature = None
        best_loss = 100
        for T in Ts:
            loss = objective_temperature(T)
            if loss<best_loss:
                best_loss = loss
                best_temperature = T

        print(f"best_temperature: {best_temperature:.4f}")
        with torch.no_grad():
            test_preds_list = []
            test_labels_list = []
            test_probs_list = []
            for batch_X, batch_y in Target_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                test_logits = model(batch_X) 
                test_logits = test_logits/best_temperature
                test_probs_all = torch.softmax(test_logits, dim=1)
                test_preds = torch.argmax(test_probs_all, dim=1)
                test_probs = test_probs_all.max(dim=1).values
                test_preds_list.append(test_preds)
                test_labels_list.append(batch_y)
                test_probs_list.append(test_probs)

        # compute ECL's ECE
        test_preds = torch.cat(test_preds_list,dim=0).cpu().numpy()
        test_labels = torch.cat(test_labels_list,dim=0).cpu().numpy()
        test_probs = torch.cat(test_probs_list,dim=0).cpu().numpy()
        ece, _, _, _, _ = equal_interval_ece(test_probs, test_preds==test_labels, num_bins=15)
        index = test_preds==test_labels
        acc = sum(index)/len(index)
        print(f"ECL's ECE: {ece:.4f}, ECL's ACC: {acc:.4f}")

    if run_Oracle_TS:
        train_logit_list = []
        Train_label_list = []
        model.eval()
        with torch.no_grad():
            for batch_X, batch_y in Target_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                train_logits = model(batch_X) 
                train_logit_list.append(train_logits)
                Train_label_list.append(batch_y)
        train_logits = torch.cat(train_logit_list,dim=0)
        train_labels = torch.cat(Train_label_list,dim=0)

        def calibration_loss(T, train_logits, Train_labels):
            train_logits = train_logits/T.item()

            train_preds = torch.argmax(train_logits, dim=1)
            train_probs = torch.softmax(train_logits, dim=1).max(dim=1).values
            Train_H = train_preds==Train_labels

            ece = ECELoss()(train_probs, Train_H)
            #print(f"T: {T.item():4f}")

            return ece

        def objective_temperature(T):
            return calibration_loss(T, train_logits, train_labels).item()
        
        initial_temperature = [30.0]
        bounds = [(1.0,30.0)]
        best_temperature = minimize(objective_temperature, initial_temperature, method='Powell',bounds=bounds,options={'disp': True}).x[0].item()

        print(f"best_temperature: {best_temperature:4f}")

        with torch.no_grad():
            test_preds_list = []
            test_labels_list = []
            test_probs_list = []
            for batch_X, batch_y in Target_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                test_logits = model(batch_X) 
                test_logits = test_logits/best_temperature
                test_probs_all = torch.softmax(test_logits, dim=1)
                test_preds = torch.argmax(test_probs_all, dim=1)
                test_probs = test_probs_all.max(dim=1).values
                test_preds_list.append(test_preds)
                test_labels_list.append(batch_y)
                test_probs_list.append(test_probs)

        # compute Oracle_TS's ECE
        test_preds = torch.cat(test_preds_list,dim=0).cpu().numpy()
        test_labels = torch.cat(test_labels_list,dim=0).cpu().numpy()
        test_probs = torch.cat(test_probs_list,dim=0).cpu().numpy()
        ece, _, _, _, _ = equal_interval_ece(test_probs, test_preds==test_labels, num_bins=15)
        index = test_preds==test_labels
        acc = sum(index)/len(index)
        print(f"Oracle_TS's ECE: {ece:.4f}")

    print("Success on "+model.name+" on "+target_domain)