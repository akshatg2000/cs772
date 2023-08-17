import torch
import torch.nn.functional as F
import pickle



class SkipGram(torch.nn.Module):
    def __init__(self, vocab_size, dim,reg_lambda=0.01):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size    #Total Vocab size of our dataset   
        self.dim = dim          #dimensions of word vector
        self.embedding_weights = torch.nn.Embedding(vocab_size, dim)  #Embedding matrix for all the words
        self.reg_lambda=reg_lambda
    
    def forward(self, target, context):
        u = self.embedding_weights(target)  #word vector for target
        v = self.embedding_weights(context) #word vector for context
        # output = torch.sum(u * v,dim=1)  #output values

        emb_product = torch.mul(u, v)            
        emb_product = torch.sum(emb_product, dim=1)   
        return emb_product
    
    def loss(self, output):
      eps=1e-10
      # loss = -torch.mean(torch.log(torch.sigmoid(output)+eps)) + self.reg_lambda * torch.sum(self.embedding_weights.weight ** 2)
      loss= -torch.mean(torch.log(torch.sigmoid(output)))
    #   loss= -F.logsigmoid(output)
    #   loss=loss.mean()
      return loss
    
    
def train(model, target_list, context_list, num_epochs, batch_size, learning_rate, reg_lambda=0.01, checkpoint_path=None, resume_training=False):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(device)
    start_epoch = 0
    
    if resume_training and checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("Loaded checkpoint from epoch", start_epoch)

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        num_batches = int(len(target_list) / batch_size)
        for batch in range(num_batches):

            start = batch * batch_size
            end = start + batch_size
            target_batch = torch.tensor(target_list[start:end], dtype=torch.long).to(device)
            context_batch = torch.tensor(context_list[start:end], dtype=torch.long).to(device)
            output = model.forward(target_batch, context_batch)
            loss = model.loss(output)
            total_loss += loss.item()
            custom_backprop(model, output, target_batch, context_batch, learning_rate, reg_lambda)
            
            if (batch+1) % 1000 == 0:
              print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch+1}/{num_batches}], Loss: {loss.item():.4f}')

        if epoch % 1 == 0:
            print("Epoch {}: Loss = {}".format(epoch, total_loss))

        # Save model and optimizer state every few epochs
        if (epoch+1) % 2 == 0 and checkpoint_path:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': loss
            }, checkpoint_path)
            print("Saved checkpoint for epoch", epoch)

    model.cpu()
      


def custom_backprop(model, output,context_batch, target_batch, learning_rate, reg_lambda):
   

    dl_du = torch.zeros_like(model.embedding_weights.weight.data[target_batch]).to(device)
    dl_dv = torch.zeros_like(model.embedding_weights.weight.data[context_batch]).to(device)
    
    for i in range(target_batch.shape[0]):
        u = model.embedding_weights.weight.data[target_batch[i]]
        v = model.embedding_weights.weight.data[context_batch[i]]
        # dl_du[i] = (torch.sigmoid(output[i]) - 1) * v + 2 * reg_lambda * u
        # dl_dv[i] = (torch.sigmoid(output[i]) - 1) * u + 2 * reg_lambda * v
        dl_du[i] = (torch.sigmoid(output[i]) - 1) * v 
        dl_dv[i] = (torch.sigmoid(output[i]) - 1) * u 
    
 

    # dl_du = torch.clamp(dl_du, -1, 1)
    # dl_dv = torch.clamp(dl_dv, -1, 1)
    
    model.embedding_weights.weight.data[target_batch] -= learning_rate * dl_du
    model.embedding_weights.weight.data[context_batch] -= learning_rate * dl_dv
    # print(model.embedding_weights.weight.data)



if __name__ == "__main__":
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('context.pkl', 'rb') as g:
        context = pickle.load(g)
    with open('target.pkl', 'rb') as h:
        target = pickle.load(h)
    
    model=SkipGram(len(vocab),300)
    
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    checkpoint_path = 'model_checkpoint.pt'
    train(model, target, context, num_epochs=80, batch_size=256, learning_rate=0.01, reg_lambda=0.01, checkpoint_path=checkpoint_path, resume_training=True)