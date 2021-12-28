import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(35813)

# Centeredness with Frobenious Distance
def frobenious_distance(mat1,mat2):
    # Utility
    # Return matrix shaped: (Views, Subjects)
    # Each View of each subject has corresponding distance calculation.
    return torch.sqrt(torch.square(torch.abs(mat1 - mat2)).sum(dim=(2,3))).transpose(1,0).to(device)

def view_normalization(views):
    # Utility
    # Return matrix shaped: (Views, Subjects)
    # Each View of each subject has different view normalizer.
    return views.mean(dim=(1,2,3)).max() / views.mean(dim=(1,2,3))

def FrobLoss(cbt, subjects, sample_size=10, aggr="sum"):
    subset = subjects[torch.randint(len(subjects), (sample_size,))].to(device)
    subset = subset.reshape((subset.size()[3],subset.size()[0],subset.size()[1],subset.size()[2]))
    if aggr == "sum":return (frobenious_distance(cbt,subset)*view_normalization(subset)).sum()
    if aggr == "mean":return (frobenious_distance(cbt,subset)*view_normalization(subset)).mean()

# def MeanFrobLoss(cbts, targets,sample_size=10):
#     # Measuring DGN Frob Loss.
#     losses = []
#     for idx, cbt in enumerate(cbts):
#         losses.append(FrobLoss(cbt, targets[:,idx,:,:,:], sample_size=sample_size, aggr="mean" ))

#     return torch.stack(losses)

def TestFrobLoss(cbts, targets, sample_size=10,alpha=0.3):
    # Separate measuring of all losses for RDGN evaluation.
    reg_loss = []
    loss=[]
    for idx,cbt in enumerate(cbts):
        loss.append(FrobLoss(cbt, targets[:,idx,:,:,:],sample_size=sample_size,aggr="mean"))
        if idx != 0: 
            reg_loss.append(alpha * torch.sqrt(torch.square((cbts - cbts[idx - 1])).sum()))
        else: continue
        
    return torch.stack(loss).to(device), torch.stack(reg_loss).to(device)

def MultiFrobLoss(cbts, targets, sample_size=10, aggr="sum",alpha=0.3):
    # Combination of all losses are measured during RDGN training.
    losses = []
    for idx,cbt in enumerate(cbts):
        # DGN --> Frob Loss is sum of distances: (views - population)
        loss = FrobLoss(cbt, targets[:,idx,:,:,:],sample_size=sample_size, aggr=aggr)
        if idx != 0: 
            loss = loss + alpha * torch.sqrt(torch.square((cbts - cbts[idx - 1])).sum())
        losses.append(loss)
    return torch.stack(losses).to(device)

if __name__=="__main__":
    samples = torch.rand(100,3,35,35,4)
    cbt = torch.rand(35,35) # cbt over samples
    sub = samples[0,0,:,:,:] #

    # loss1 = MeanFrobLoss([cbt],samples)
    # print(loss1)

    # cbts = torch.rand(3,35,35)
    # loss2 = MeanFrobLoss(cbts,samples)
    # print(loss2)