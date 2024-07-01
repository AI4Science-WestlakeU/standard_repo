#general imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import pdb
import os
import sys
import time
import argparse
import wandb
import tqdm

#import path
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))

#custom imports
import project_module.utils.utils as utils
from project_module.filepath import EXP_PATH,CURRENT_WP
from project_module.model.model_demo import Net
class Trainner(object):
    def __init__(self,args,model,train_dataloader,test_dataloader):
        self.args=args
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.optimizer = self.get_optimizer()
        self.loss_fn = self.get_loss_fn()
        self.train_dataloader=train_dataloader
        self.test_dataloader=test_dataloader

    def get_optimizer(self):
        optimizer = None
        return optimizer
    def get_scheduler(self):
        scheduler = None
        return scheduler

    def get_loss_fn(self):
        loss_fn = None
        return loss_fn
    def train(self):
        args=self.args
        optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)
        wandb.watch(self.model)
        
        global_step = 0
        for epoch in tqdm.trange(1, args.epochs + 1):
            # training
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_dataloader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                global_step +=1
                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_dataloader.dataset),batch_idx / len(train_dataloader), loss.item()))
                    training_log = {
                        "train/loss": loss.detach().item(),
                    }
                # wandb 可视化训练损失（此处仅展示保存训练曲线）
                wandb.log(training_log, step=global_step)
            self.model.eval()
            test_loss = 0
            correct = 0

            example_images = []
            with torch.no_grad():
                for data, target in test_dataloader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    # sum up batch loss
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    # get the index of the max log-probability
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    example_images.append(wandb.Image(
                        data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

            test_loss /= len(test_dataloader.dataset)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0%})\n'.format(
                test_loss, correct, len(test_dataloader.dataset),
                correct / len(test_dataloader.dataset)))
            
            test_log = {
                "Test/Examples": example_images, # 保存测试图像
                "Test/Accuracy": 100. * correct / len(test_dataloader.dataset), 
                "Test/Loss": test_loss}
            # 保存test可视化内容
            wandb.log(test_log, step=global_step)
            if epoch%self.args.save_interval==0:
                pdb.set_trace()
                if not os.path.exists(self.args.results_path):
                    os.makedirs(self.args.results_path)
                model_path = os.path.join(self.args.results_path,f'model_{epoch}.pth')
                # torch.save(model_path, self.model.state_dict())
                torch.save(self.model.state_dict(), model_path)
        wandb.finish()
    
    def validate(self):
        pass
    
    def test(self):
        pass

if __name__=='__main__':
    # config the args
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default='/zhangtao/general/project_module/configs/config.yaml')
    parser = utils.add_args_from_config(parser)
    args=parser.parse_args()
    args.exp_path=EXP_PATH
    args.date_time = time.strftime('%Y%m%d_%H%M%S')
    args.results_path = os.path.join(args.exp_path,'results',args.date_time,args.exp_name,'train')
    utils.save_config_from_args(args)
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    # config model
    model=Net().to(device)
    
    # config dataloader 
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.train_batch_size, shuffle=True, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    # config wandb
    if args.use_wandb:
        os.environ["WANDB_API_KEY"] = "88b926fd6325fdc9ab9afc2292d8fe2d2664951a"
        wandb.login()
        wandb.init(project=args.wandb_project, config=args)
    utils.set_seed(args.seed)
    # config trainner 
    trainner=Trainner(args,model,train_dataloader,test_dataloader)
    trainner.train()