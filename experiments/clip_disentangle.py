import torch
import clip
from models.base_model import CLIPDomainDisentangleModel
from torch import nn
from torch.nn import functional as F
from torch import cat
import numpy as np


class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = 1.0 * b.sum()
        return b / x.size(0)

class CLIPDisentangleExperiment:
    
    def __init__(self, opt):
        # Utils
        self.opt = opt
        self.device = torch.device('cpu' if  not torch.cuda.is_available() else 'cuda:0')

        # Setup model
        self.model = CLIPDomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        self.clip_model, _ = clip.load('ViT-B/32', device='cpu') # load it first to CPU to ensure you're using fp32 precision.
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Setup optimization procedure and losses
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.domain_loss = torch.nn.CrossEntropyLoss()
        self.class_loss = torch.nn.CrossEntropyLoss()
        self.reconstructor_loss = torch.nn.MSELoss()
        self.clip_loss = torch.nn.MSELoss()
        self.class_loss_ent = EntropyLoss()
        self.domain_loss_ent = EntropyLoss() 

    def train_iteration(self, data):
        src_img, src_y, src_d, trg_img, _, trg_d = data
    
        src_with_desc_mask = [d != "" for d in src_d]
        trg_with_desc_mask = [d != "" for d in trg_d]
        src_img = src_img.to(self.device)
        src_y = src_y.to(self.device)
        src_desc = clip.tokenize(src_d, truncate = True).to(self.device)[src_with_desc_mask]
        src_desc = self.clip_model.encode_text(src_desc)

        trg_img = trg_img.to(self.device)
        trg_desc = clip.tokenize(trg_d, truncate = True).to(self.device)[trg_with_desc_mask]
        trg_desc = self.clip_model.encode_text(trg_desc)
        
        self.optimizer.zero_grad()

        # Processing a Source Domain Image
        src_class_output, src_domain_output, src_features, src_reconstructed_features, src_class_output_ds, src_domain_output_cs, src_f_ds = self.model(src_img)
        _, trg_domain_output, trg_features, trg_reconstructed_features, trg_class_output_ds, trg_domain_output_cs, trg_f_ds = self.model(trg_img)

        # source class loss
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

        #CLIP loss
        if not any(src_with_desc_mask):
            src_clip_loss = 0
        else:
            src_clip_loss = self.clip_loss(src_desc, src_f_ds[src_with_desc_mask])
        if not any(trg_with_desc_mask):
            trg_clip_loss = 0
        else:
            trg_clip_loss = self.clip_loss(trg_desc, trg_f_ds[trg_with_desc_mask])

        tot_clip_loss = (trg_clip_loss + src_clip_loss) / 2

        tot_loss_domain_ent = self.domain_loss_ent(cat((src_domain_output_cs, trg_domain_output_cs), dim=0))
 
        tot_loss = (0.4 * src_loss_class) + (0.08 * tot_loss_domain) + (0.02 * tot_loss_rec) + (0.4 * src_loss_class_ent) + (0.08 * tot_loss_domain_ent) + (0.02 * tot_clip_loss)

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