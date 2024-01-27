import torch
import torch.nn as nn
from loss import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss
    


def selectedLoss(output,y,selection_out,num_classes):
  # print(y)
  sum_ = 0
  size = y.size()[0]
  # softMax = torch.nn.Softmax(dim=1) #分部实现交叉熵的第一步 进行softmax
  # decrementSum = torch.nn.NLLLoss()
  # output_temp = softMax(output)
  for i in range(0,size):
    tip = torch.max(torch.tensor([0.]).to(device),(selection_out-selection_out[i])+torch.tensor([1]).to(device)) #公式20的 w的实现，每一个样本对应一个乘数
    temp = tip * CrossEntropyLabelSmooth(reduction='none', num_classes=num_classes,
                                                     epsilon=0.1)(output, y)
    sum_ = sum_ + temp.mean()
    # print(sum_.item(),end=" ")
  sum_ = sum_/(torch.tensor(size,dtype = torch.float).to(device))
  return sum_

def selectedLoss1(output,target,selection_out):
  sum_ = 0
  size = output.size()[0]
  for i in range(0,size):
    tip = torch.max(torch.tensor([0.]).to(device),(selection_out-selection_out[i])+torch.tensor([1]).to(device)) #公式20的 w的实现，每一个样本对应一个乘数
    tempOutput = tip * output
    # print(tip)
    loss_ = MMDLoss(kernel_type='rbf')(tempOutput,target)#其实有点不一样，就是，原论文是对每个样本计算损失的，这个是对一部分源域和目标域计算的
    sum_ = sum_ + loss_
    # print(sum_.item(),end=" ")
  sum_ = sum_/(torch.tensor(size,dtype = torch.float).to(device))
  return sum_


#slmmd(source, target, s_label, t_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
def selectedLoss2(output,target,s_label, t_label,selection_out):
  sum_ = 0
  size = output.size()[0]
  for i in range(0,size):
    tip = torch.max(torch.tensor([0.]).to(device),(selection_out-selection_out[i])+torch.tensor([1]).to(device)) #公式20的 w的实现，每一个样本对应一个乘数
    tempOutput = tip * output
    # print(tip)
    loss_ = slmmd(tempOutput,target,s_label, t_label)
    sum_ = sum_ + loss_
    # print(sum_.item(),end=" ")
  sum_ = sum_/(torch.tensor(size,dtype = torch.float).to(device))
  return sum_


