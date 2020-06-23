import pickle ,argparse, torch
from tqdm import tqdm
from models import EncoderCNN, DecoderRNN

def get_args():
    p = argparse.ArgumentParser()
    
    p.add_argument('project_path', type=str,
                   help="Path to the project")
    
    args = p.parse_args()
    
    return args

def load_models(project_path):
    
    vocab = pickle.load(open('{}vocab/vocab.pkl'.format(project_path), 'rb'))
    
    print('Loading Encoder and Decoder from {}'.format('{}trained_models'.format(project_path)))
          
    encoder = EncoderCNN(300)
    encoder.load_state_dict(torch.load(('{}trained_models/encoder.pt'.format(project_path))))
    
    print('Encoder Loaded')
    
    decoder = DecoderRNN(embed_size=300, hidden_size=512, vocab_size=len(vocab))
    decoder.load_state_dict(torch.load(('{}trained_models/decoder.pt'.format(project_path))))
    
    print('Decoder Loaded')
    
    return encoder, decoder, vocab

def get_dataloader(project_path):
    
    testing_dataloader = pickle.load(open('{}dataloaders/testing_dataloader.pkl'.format(project_path), 'rb'))
    
    return testing_dataloader

def decode_text(list_nums, vocab):
    decode_vocab = {num: word for word, num in vocab.items()}
    
    
    decoded = [decode_vocab[i] for i in list_nums]
    
    cleaned = ' '.join([i for i in decoded if i not in ['<start>', '<pad>', '<end>']])

    return cleaned

def get_test_obj(dataset_tuple):

    test_image = dataset_tuple[0]
    caption = dataset_tuple[2]
    
    return test_image, caption

def test(encoder, decoder, dataloader, vocab):
    dev = 'cuda'
    
    encoder.eval()
    decoder.eval()

    encoder.to(dev)
    decoder.to(dev)
    
    titles = []
    
    for i in tqdm(dataloader):

        try:
            image = i[0].to(dev)
            caption = i[1].to(dev)
            features = encoder(image)
            output = decoder.sample(features.unsqueeze(1), 4)
            cleaned_text = decode_text(output, vocab)
            titles.append(cleaned_text)

        except:

            continue
            
    return titles
      
def main(args):
    encoder, decoder, vocab = load_models(args.project_path)
    testing_dataloader = get_dataloader(args.project_path)
    predictions = test(encoder, decoder, testing_dataloader, vocab)
    print(list(set(predictions)))
    
if __name__ == '__main__':
    args = get_args()
    main(args)
    
    