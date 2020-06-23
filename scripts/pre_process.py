import pandas as pd, numpy as np, torch, argparse
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence,pad_packed_sequence
from torch.utils.data import DataLoader
import torch, os, pickle
from image_dataset import CustomDataset

pre_process = transforms.Compose([ 
                  transforms.Resize(256),                         
                  transforms.RandomCrop(224),                      
                  transforms.RandomHorizontalFlip(),               
                  transforms.ToTensor(),                           
                  transforms.Normalize((0.485, 0.456, 0.406),      
                                      (0.229, 0.224, 0.225))
                                      ])

def get_args():

    p = argparse.ArgumentParser()

    p.add_argument('data_dir', type=str,
                   help="Data folder that contains artworks")
    
    p.add_argument('meta_data', type=str,
                    help="Csv file containing meta data")
    
    p.add_argument('output_dir', type=str, default = None,
                    help="Directory to save resized images")

    p.add_argument('--dataset_size', type=int, default=None,
                    help="Max number of items in dataset")

    p.add_argument('--batch_size', type=int, default=100,
                   help="Batch size for training data")
    return p.parse_args()

def generate_dataframe(art_type, df):
    data = {'TITLE' : [], 'FILE' : [], 'TYPE' : []}
    
    if art_type in set(df['TYPE']):
        
        for title, i, file in zip(df['TITLE'], df['TYPE'], df['FILE']):
        
            if i == art_type:
                data['TITLE'].append(title.lower())
                data['FILE'].append(file)
                data['TYPE'].append(i)
 
    else:
        return print('Wrong Type try again')
 
    return pd.DataFrame(data=data)

def reconstruct_image(input_image):
    x = input_image
    z = x * torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    z = z + torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    img2 = transforms.ToPILImage(mode='RGB')(z)
    return img2

def create_folders(args):
    
    base_dir = args.output_dir
    data_folder = '{}data/'.format(base_dir)
    
    folders = {
        'base_dir'          : args.output_dir,
        'data_folder'       :'{}data/'.format(base_dir),
        'training_images'   :'{}training_images/'.format(data_folder),
        'testing_images'    :'{}testing_images/'.format(data_folder),
        'dataloader_folder' : '{}dataloaders/'.format(base_dir),
        'vocab_folder'      : '{}vocab/'.format(base_dir),
        'meta_data'  : '{}meta_data/'.format(base_dir),
        'trained_models'    : '{}trained_models'.format(base_dir)
    }
        
    for path in folders.values():
        if os.path.exists(path) == False:
            os.mkdir(path)
            
    return folders
    
def resize_images(args, folders, output, pretrained, test_perc):
    
  
    image_folder = folders['training_images']
    test_image_folder = folders['testing_images']
    
    pre_processed_images = []
    processed_images = []
    captions = []
    image_paths = []
    failed_path = []
    artwork_type = []
    data = {'FILE' : [], 'TITLE' : [], 'TYPE' : []}
    
    print('Processing Artwork\n')

    df = pd.read_csv(args.meta_data, sep='\t', nrows=args.dataset_size)
    
    for num, i in tqdm(enumerate(zip(df['FILE'], df['TITLE'], df['TYPE'])), total=args.dataset_size):
        
        if num == (args.dataset_size - (args.dataset_size * test_perc)):
            
            training_data = (processed_images,captions, image_paths, 
                             pre_processed_images, artwork_type)
            
            pre_processed_images = []
            processed_images = []
            captions = []
            image_paths = []
            failed_path = []
            artwork_type = []
            
            pd.DataFrame(data=data).to_csv('{}training_data.csv'.format(folders['meta_data']), index=False)
            image_folder = test_image_folder
            data = {'FILE' : [], 'TITLE' : [], 'TYPE' : []}
            
        if num == args.dataset_size:
            break
            
        try:
            
            data['FILE'].append(i[0])
            data['TITLE'].append(i[1])
            data['TYPE'].append(i[2])
            
            image = Image.open('{}{}'.format(args.data_dir, i[0]))
            resized_image = pre_process(image)
            pre_processed_images.append(image)
            processed_images.append(resized_image)
            
            captions.append(i[1])
            artwork_type.append(i[2])
            
            if args.output_dir and output == True:
                
                img2 = reconstruct_image(resized_image)
                img2.save('{}{}'.format(image_folder, i[0]))
                image_paths.append('{}{}'.format(image_folder,i[0]))

            else:

                image_paths.append('{}{}'.format(args.data_dir,i[0]))

        except Exception as e:
            #print('something Wrong: {}'.format(e))
            failed_path.append('{}{}'.format(args.data_dir,i[0]))

            continue
    
    
    testing_data = (processed_images,captions, image_paths, pre_processed_images, artwork_type)
    
    num_training = len([i for i in os.listdir(folders['training_images'])])
    num_testing = len([i for i in os.listdir(folders['testing_images'])])
    
    pd.DataFrame(data=data).to_csv('{}testing_data.csv'.format(folders['meta_data']), index=False)
   
    print('\n{} artworks added to dataset'.format(num_training + num_testing))
    print('{} failed to load\n'.format(len(failed_path)))
    print('{} Training'.format(num_training))
    print('{} Testing'.format(num_testing))
        
    return training_data, testing_data

def pre_process_captions(captions, folders, train,vocab=None):
    
    tokenized_titles = [tokenize(i.lower()) for i in captions]
    
    if train==True:
        
        vocab = gen_vocab(tokenized_titles) 
        
        
        print('\nOutputting vocab object to {}{}'.format(folders['vocab_folder'], 'vocab.pkl'))
        with open('{}/vocab.pkl'.format(folders['vocab_folder']), 'wb') as file:
            pickle.dump(vocab, file)
        
    
    encoded_titles, title_lengths = encode(tokenized_titles, vocab)
    
    return (encoded_titles, tokenized_titles), vocab

def tokenize(title_string):
    return title_string.split(' ')

def gen_vocab(tokenized_data):
    
    vocab = ['<pad>', '<start>', '<end>', '<unk>']
    print('Generating Vocab\n')
    for title in tqdm(tokenized_data):
        
        for token in title:
            
            if token not in vocab:
                vocab.append(token)
            else:
                continue
    
    print('\n{} tokens in vocab'.format(len(vocab)))

    idx2wrd = dict(enumerate(set(vocab)))
    wrd2idx = {wrd : num for num, wrd in idx2wrd.items()}
    
    return wrd2idx 

def encode(titles, vocab):
    
    encoded_titles = []
    title_lengths = []
    
    for title in titles:
        title_lengths.append(len(title))
        encodings = [vocab['<start>']]
        
        for token in title:
            
            if token in vocab:
                encodings.append(vocab[token])
                
            else:
                encodings.append(vocab['<unk>'])
        
        encodings.append(vocab['<end>'])
        encoded_titles.append(torch.LongTensor(encodings))
    
    print(len(title_lengths))
    print(len(encoded_titles))
    
    return pad_sequence(encoded_titles, batch_first=True, padding_value=vocab['<pad>']), title_lengths
 
def gen_dataloader(data, captions, folders, batch_size, mode):
    
    dataloader_folder = folders['dataloader_folder']
    
    if mode == 0:
        output_path = '{}training_dataloader.pkl'.format(dataloader_folder)
        batch_size = batch_size
    elif mode == 1:
        output_path = '{}testing_dataloader.pkl'.format(dataloader_folder)
        batch_size = 1
        
    if os.path.exists(dataloader_folder) == False:
        os.mkdir(dataloader_folder)
        
    dataset = CustomDataset(images=data[0], captions=captions, image_paths=data[2], 
                                  raw_captions=data[1])
    
    dataloader = DataLoader(dataset, batch_size = batch_size)
    print('\nOutputting dataloader object to {}{}'.format(dataloader_folder, output_path))
    
    with open(output_path, 'wb') as file:
        pickle.dump(dataloader, file)
    
    return dataloader
            
def main():
    print('\nStarting...\n')
    args = get_args()
    folders = create_folders(args)
    training_data, testing_data = resize_images(args, folders, output=True, pretrained=True,
                                        test_perc=0.2)
    training_captions, vocab = pre_process_captions(training_data[1], folders, train=True)
    
    d_v = {num : word for word, num in vocab.items()}
    testing_captions, vocab = pre_process_captions(testing_data[1], folders, train=False, vocab=vocab)
    training_dataloader = gen_dataloader(training_data, training_captions[0],
                                     folders,args.batch_size, 0)
    testing_dataloader = gen_dataloader(testing_data, testing_captions[0],
                                     folders, args.batch_size, 1)
    
    print('\nDone\n')
    
    return training_dataloader, testing_dataloader
    
   
if __name__ == '__main__':
    main()