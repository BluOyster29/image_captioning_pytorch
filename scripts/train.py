import pickle, argparse
from scripts.models import EncoderCNN, DecoderRNN
import torch, math
from tqdm.notebook import tqdm
import torch.nn as nn

def get_args():

    p = argparse.ArgumentParser()
    p.add_argument('project_path', type=str,
                    help='The folder where everything is kept')

    p.add_argument('--num_epochs', type=int, default=10,
                    help="Number of training epochs")

    args = p.parse_args()

    return args

def load_objects(project_path):

    vocab_path = '{}vocab/vocab.pkl'.format(project_path)
    dataloader_path = '{}dataloader/dataloader.pkl'.format(project_path)

    print('Loading vocab')
    with open(vocab_path, 'rb') as file:
        vocab = pickle.load(file)

    print('Loading Dataloader')
    with open(dataloader_path, 'rb') as file:
        dataloader = pickle.load(file)
    
    print('Done!')
    return vocab, dataloader

def train(train_dataloader, args, vocab, num_epochs):

    encoder_model = EncoderCNN(300)
    decoder_model = DecoderRNN(embed_size=300, hidden_size=512, vocab_size=len(vocab))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_model.to(device)
    decoder_model.to(device)
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

    params = list(decoder_model.parameters()) + list(encoder_model.embed.parameters())

    optimizer = torch.optim.Adam(params = params, lr = 0.001)

    total_step = math.ceil(len(train_dataloader.dataset.caption_lengths) / train_dataloader.batch_sampler.batch_size)

    optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.001)

    encoder_model.train()
    decoder_model.train()
    vocab_size = len(vocab)
    num = 1

    for epoch in tqdm(range(0, num_epochs),total=num_epochs):
        
        
        
        for i in tqdm(train_dataloader):
            
            
            image = i[0].to(device)
            caption = i[1].to(device)

            decoder_model.zero_grad()
            encoder_model.zero_grad()


            features = encoder_model(image)
            outputs = decoder_model(features, caption)

            loss = criterion(outputs.view(-1, vocab_size), caption.view(-1))
            loss.backward()
            optimizer.step()

            num+=1 

        print('Loss after epoch {}: {}'.format(epoch, loss))

    return encoder_model, decoder_model

def output_model(trained_model, path):

    output_path = '{}/trained_model/model.pt'.format(path)
    torch.save(trained_model.state_dict(), output_path)
    
def main():
    args = get_args()
    print('Loading vocab and dataloader from {}'.format(args.project_path))
    vocab, dataloader = load_objects(args.project_path)
    print('\nTraining...')
    trained_encoder, trained_decoder = train(dataloader, args, vocab)
    print('\nSaving model to {}/trained_model/'.format(args.project_path))


if __name__ == '__main__':
    main()