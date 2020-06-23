import pickle, argparse
from models import EncoderCNN, DecoderRNN
import torch, math
from tqdm import tqdm
import torch.nn as nn

def get_args():

    p = argparse.ArgumentParser()
    p.add_argument('project_path', type=str,
                    help='The folder where everything is kept')

    p.add_argument('--num_epochs', type=int, default=10,
                    help="Number of training epochs")

    args = p.parse_args()

    return args

def gen_folders(project_dir):
    
    folders ={
        'trained_model': '{}trained_model/'.format(project_dir),
        'training_stats' : '{}stats/'.format(project_dir)
    }
    
    return folders
    
def load_objects(project_path):
    print('Loading Vocab from {}vocab/vocab.pkl'.format(project_path))
    
    vocab = pickle.load(open('{}vocab/vocab.pkl'.format(project_path), 'rb'))

    print('Loading Dataloaders from {}dataloaders/'.format(project_path))
    
    training_dataloader = pickle.load(open('{}dataloaders/training_dataloader.pkl'.format(project_path), 'rb'))
    
    print('Training Loaded')
    testing_dataloadr = pickle.load(open('{}dataloaders/testing_dataloader.pkl'.format(project_path), 'rb'))
    
    return vocab, training_dataloader
  
def train(train_dataloader, args, vocab):

    print('Loading Models')
    encoder_model = EncoderCNN(300)
    decoder_model = DecoderRNN(embed_size=300, hidden_size=512, vocab_size=len(vocab))

    #device = 'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    encoder_model.to(device)
    decoder_model.to(device)
    
    print('Setting Parameters')
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    params = list(decoder_model.parameters()) + list(encoder_model.embed.parameters())
    total_step = math.ceil(len(train_dataloader.dataset.caption_lengths) / train_dataloader.batch_sampler.batch_size)

    optimizer = torch.optim.Adam(encoder_model.parameters(), lr=0.001)

    encoder_model.train()
    decoder_model.train()
    
    vocab_size = len(vocab)
    num = 1
    failed_batch = 0
    
    print('Training {} batches of {} samples over {} epochs'.format(len(train_dataloader), train_dataloader.batch_sampler.batch_size, args.num_epochs))
    for epoch in tqdm(range(1, args.num_epochs+1),total=args.num_epochs):
         
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

def output_model(encoder, decoder, path):

    encoder_path = '{}trained_models/encoder.pt'.format(path)
    print('Saving Encoder to {}'.format(encoder_path))
    torch.save(encoder.state_dict(), encoder_path)
   
    decoder_path = '{}trained_models/decoder.pt'.format(path)
    print('Saving Decoder to {}'.format(decoder_path))
    torch.save(decoder.state_dict(), decoder_path)
    
    
    
def main():
    args = get_args()
    vocab, training_dataloader = load_objects(args.project_path)
    print('\nTraining...')
    trained_encoder, trained_decoder = train(training_dataloader, args, vocab)
    output_model(trained_encoder,trained_decoder, args.project_path)


if __name__ == '__main__':
    main()