import torch
# from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F

def create_dataset(input_path):
    """Read data from input_path""" 
    with open(input_path, 'r') as f:
        data = f.read()
    words = data.splitlines()
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    vocab = sorted(list(set(''.join(words))))
    max_length = max(len(w) for w in words)
    return words, vocab, max_length

class CharDataset(Dataset):
    def __init__(self, words, vocab, max_length) -> None:
        
        self.words = words
        self.vocab = vocab
        self.max_length = max_length
        self.stoi = {s: i+1 for i,s in enumerate(vocab)}
        self.itos = {s: i for i,s in self.stoi.items()}

    def __len__(self):
        return len(self.words)

    def encode(self, name): 
        return torch.tensor([self.stoi[n] for n in name], dtype = torch.long)
    
    def decode(self, tokens): 
        return ''.join(self.itos[t] for t in tokens)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        x = torch.zeros(self.max_length+1, dtype=torch.long)
        y = torch.zeros(self.max_length+1, dtype=torch.long)
        ix = self.encode(word)
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1
        return x, y

@torch.inference_mode()
def evaluate_loss(model, tr_loader, te_loader, device, num_batches = 10):
    model.eval()
    loss_tr = []
    loss_te = []
    for n in range(num_batches):
        Xtr, Ytr = next(iter(tr_loader))
        Xte, Yte = next(iter(te_loader))
        Xtr = Xtr.to(device)
        Ytr = Ytr.to(device)
        Xte = Xte.to(device)
        Yte = Yte.to(device)
        _, train_loss = model(Xtr, Ytr)
        _, test_loss = model(Xte, Yte)
        loss_tr.append(train_loss)
        loss_te.append(test_loss)
    
    mean_train_loss = torch.tensor(loss_tr).mean().item()
    mean_test_loss = torch.tensor(loss_te).mean().item()
    model.train()
    return(mean_train_loss, mean_test_loss)

@torch.no_grad()
def _generate(model, idx, max_new_tokens, device, block_size=16):
    """Generates a single batch of names based on since of idx matrix. Accessed via print_samples"""
    for _ in range(max_new_tokens):
        # print('idx shape:',idx.shape)
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        idx_cond = idx_cond.to(device)
        logits, _ = model(idx_cond)
        # Pick only the logits from most recent time step. Karpathy also does a divide by temp?
        # This is just Platt scaling which makes the various Softmax curves closes adding more randomness
        # see scratch.ipynb. https://en.wikipedia.org/wiki/Platt_scaling
        logits = logits[:,-1,:]
        probs = F.softmax(logits, dim=-1)
        # print('prob dist:',probs)
        idx_next = torch.multinomial(probs, num_samples=1)
        # print('idx_next shape:',idx_next.shape)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def print_samples(model, train_data, max_new_tokens, device, num=10):
    """ samples from the model and pretty prints the decoded samples """
    X_init = torch.zeros((num, 1), dtype=torch.long).to(device)
    X_samp = _generate(model, X_init, max_new_tokens, device)[:,1:].tolist()
    # print(X_samp)
    for row in X_samp:
        crop_index = row.index(0) if 0 in row else len(row)
        # print(row, crop_index)
        row = row[:crop_index]
        print(train_data.decode(row))

def get_lr_loss(model, optimizer, train_dataloader, batch_size, num_epochs, lr_start_exp=-3, lr_end_exp=0.5):

    lrexp = torch.linspace(lr_start_exp, lr_end_exp, num_epochs, requires_grad=False)
    lrs_val = 10**lrexp

    lri = []
    lossi = []
    # Training loop with mini-batches and lr sweep
    for epoch in range(num_epochs):

        ## Set learning rate
        for g in optimizer.param_groups:
            g['lr'] = lrs_val[epoch]

        xb, yb = next(iter(train_dataloader))
        # print(xb.shape, yb.shape)

        # ix = torch.randint(0, xb.shape[0], (batch_size,))

        # inputs = xb[ix]
        # targets = yb[ix]

        # Forward pass
        _, loss = model(xb, yb)
        lri.append(lrs_val[epoch])
        lossi.append(loss.item())
        # print(loss.item())
        # loss = loss_function(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return lri, lossi


##TODO: 4. Separate tokenizer class so you dont need to pass dataset to decode
##TODO: 5. Maybe start plotting using Bokeh or Altair
##TODO: 6. Utilities to save model checkpoints during training