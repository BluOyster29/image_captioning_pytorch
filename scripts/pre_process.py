import pandas as pd, numpy as np, torch, argparse
from PIL import Image
from tqdm.notebook import tqdm
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence,pad_packed_sequence
from torch.utils.data import DataLoader
import torch, os, pickle
from scripts.image_dataset import CustomDataset

def get_args():

    p = argparse.ArgumentParser()

    p.add_argument('data_dir', type=str,
                   help="Data folder that contains artworks")

    p.add_argument('--output_dir', type=str, default = None,
                    help="Directory to save resized images")

    p.add_argument('meta_data', type=str,
                    help="Csv file containing meta data")

    p.add_argument('--dataset_size', type=int, default=None,
                    help="Max number of items in dataset")

    p.add_argument('--batch_size', type=int, default=100,
                   help="Batch size for training data")

    p.add_argument('--vocab_dir', type=str, default='data/vocab/',
                   help="Directory to save vocab object")

    p.add_argument('--dataloader_dir', type=str, default='data/dataloaders/',
                   help="Directory to save dataloader object")

    return p.parse_args()

def pre_process(image, pretrained):

    if pretrained == True:
        pre_process = transforms.Compose([ 
                      transforms.Resize(256),                          # smaller edge of image resized to 256
                      transforms.RandomCrop(224),                      # get 224x224 crop from random location
                      transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5

                      transforms.ToTensor(),                           # convert the PIL Image to a tensor
                      transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                          (0.229, 0.224, 0.225))
                                          ])

    elif pretrained == False:
        pre_process = transforms.Compose([ 
                      transforms.Resize(224),
                      transforms.ToTensor().
                      transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                                          (0.229, 0.224, 0.225))
                                          ])
        
    return pre_process(image)

def reconstruct_image(input_image):
    x = input_image
    z = x * torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    z = z + torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    img2 = transforms.ToPILImage(mode='RGB')(z)
    return img2

def resize_images(args, output):

    df = pd.read_csv(args.meta_data, sep='\t')

    if os.path.exists(args.output_dir) == False:
        os.mkdir(args.output_dir)

        image_folder = '{}images/'.format(args.output_dir)
        os.mkdir(image_folder)


    image_folder = '{}images/'.format(args.output_dir)
    
    images = []
    captions = []
    image_paths = []
    failed_path = []
    
    print('Processing Artwork\n')

    for num, i in tqdm(enumerate(zip(df['FILE'], df['TITLE'])), total=args.dataset_size):
        
        if num == args.dataset_size:
            break
            
        try:
            resized_image = pre_process(Image.open('{}{}'.format(args.data_dir, i[0])))
            images.append(resized_image)

            captions.append(i[1])
            
            if args.output_dir and output == True:
                
                img2 = reconstruct_image(resized_image)
                img2.save('{}{}'.format(image_folder, i[0]))
                image_paths.append('{}{}'.format(image_folder,i[0]))

            else:

                image_paths.append('{}{}'.format(args.data_dir,i[0]))

        except Exception as e:
            print('something Wrong: {}'.format(e))
            failed_path.append('{}{}'.format(args.data_dir,i[0]))

            continue
        
    print('\n{} artworks added to dataset'.format(len(images)))
    print('{} failed to load\n'.format(len(failed_path)))

    return images,captions, image_paths

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
    
    return pad_sequence(encoded_titles, batch_first=True, padding_value=vocab['<pad>']), title_lengths

def pickle_data(object, path, keyword):

    output_folder = '{}{}/'.format(path,keyword)
    
    if os.path.exists(output_folder) == False:
        os.mkdir(output_folder)

    output_path = '{}{}.pkl'.format(output_folder, keyword)

    with open(output_path, 'wb') as file:
        pickle.dump(object, file)

    return output_path

def output_csv(images, captions, image_paths, path):
    data = {
        'id' : image_paths,
        'captions' : captions,
    }

    meta_data = '{}meta_data.csv'.format(path)

    df = pd.DataFrame(data=data).to_csv(meta_data, index=False)

    print('Csv outputted to {}'.format(meta_data))
    
def main():
    args = get_args()
    images, captions, image_paths = resize_images(args)
    tokenized_titles = [tokenize(i) for i in captions]
    vocab = gen_vocab(tokenized_titles) 
    output_csv(images,captions,image_paths, args.output_dir)
    print('\nOutputting vocab object to {}'.format(pickle_data(vocab, args.output_dir, 'vocab')))
    encoded_titles, title_lengths = encode(tokenized_titles, vocab)
    dataset = CustomDataset(images=images, captions=encoded_titles, image_paths=image_paths)
    train_dataloader = DataLoader(dataset, batch_size = args.batch_size)
    print('\nOutputting dataloader object to {}'.format(pickle_data(train_dataloader, args.output_dir, 'dataloader')))

    return train_dataloader
if __name__ == '__main__':
    main()