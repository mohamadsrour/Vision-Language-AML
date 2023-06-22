import torch
from models.base_model import DomainDisentangleModel
from torch import nn
from torch.nn import functional as F
from torch import cat
import numpy as np
import wandb






class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = 1.0 * b.sum()
        return b / x.size(0)

class DomainDisentangleExperiment:
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if  not torch.cuda.is_available() else 'cuda:0')

        if (opt['w']):
            self.W = opt['w']
        else:
            self.W = [0.04, 0.09, 0.02, 1]

        wandb.init(
            # set the wandb project where this run will be logged
            project="domain-disentangle",
            
            # track hyperparameters and run metadata
            config={
            "experiment": "domain_disentangle",
            "learning_rate": opt['lr'],
            "dataset": "PACS",
            "w_1": self.W[0],
            "w_2": self.W[1],
            "w_3": self.W[2],
            "alpha": self.W[3]
            }
        )
        # Setup model
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # Setup optimization procedure and losses
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.domain_loss = torch.nn.CrossEntropyLoss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        self.reconstructor_loss = torch.nn.MSELoss()
        self.class_loss_ent = EntropyLoss()
        self.domain_loss_ent = EntropyLoss() 

    def train_iteration(self, data):
        src_img, src_y, trg_img, _ = data
 
        src_img = src_img.to(self.device)
        src_y = src_y.to(self.device)
        trg_img = trg_img.to(self.device)
        
        self.optimizer.zero_grad()

        # Processing a Source Domain Image
        src_class_output, src_domain_output, src_features, src_reconstructed_features, src_class_output_ds, src_domain_output_cs = self.model(src_img)
        _, trg_domain_output, trg_features, trg_reconstructed_features, trg_class_output_ds, trg_domain_output_cs = self.model(trg_img)

        src_loss_class = self.class_loss(src_class_output, src_y)

        src_domain_label = torch.zeros(src_img.shape[0]).long().to(self.device)
        trg_domain_label = torch.ones(trg_img.shape[0]).long().to(self.device)

        tot_loss_domain = self.domain_loss(cat((src_domain_output, trg_domain_output), dim=0), cat((src_domain_label, trg_domain_label), dim=0))

        # source reconstructor loss
        src_loss_rec = self.reconstructor_loss(src_reconstructed_features, src_features)
        # target reconstructor loss
        trg_loss_rec = self.reconstructor_loss(trg_reconstructed_features, trg_features)

        tot_loss_rec = (src_loss_rec + trg_loss_rec) / 2

        # entropy loss of class output w.r.t. domain specific features
        src_loss_class_ent = self.class_loss_ent(src_class_output_ds)

        tot_loss_domain_ent = self.domain_loss_ent(cat((src_domain_output_cs, trg_domain_output_cs), dim=0))
        
        #tot_loss = (0.4 * src_loss_class) + (0.09 * tot_loss_domain) + (0.02 * tot_loss_rec) + (0.4 * src_loss_class_ent) + (0.09 * tot_loss_domain_ent)
        L_class = src_loss_class + self.W[3] * src_loss_class_ent
        L_domain = tot_loss_domain + self.W[3] * tot_loss_domain_ent
        L_rec = tot_loss_rec

        tot_loss = self.W[0] * L_class + self.W[1] * L_domain + self.W[2] * L_rec

        wandb.log({"L_class_ce": src_loss_class,
                   "L_class_ent": src_loss_class_ent,
                   "L_class": L_class,
                   "L_domain_ce": tot_loss_domain,
                   "L_domain_ent": tot_loss_domain_ent,
                   "L_domain": L_domain,
                   "L_rec": L_rec,
                   "L": tot_loss})
        
        tot_loss.backward()
        self.optimizer.step()
        return tot_loss.item()

    # move the checkpoint methods in an abstract class
    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}

        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss

        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)

        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        return iteration, best_accuracy, total_train_loss

    def validate(self, loader):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)[0]
                loss += self.class_loss(logits, y)
                pred = torch.argmax(logits, dim=-1)

                accuracy += (pred == y).sum().item()
                count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss