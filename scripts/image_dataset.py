from torch.utils.data import Dataset
import torch 

class CustomDataset(Dataset):
    def __init__(self, images, captions, image_paths):  

        self.images  = images
        self.captions     = captions
        self.caption_lengths = torch.LongTensor([len(i) for i in self.captions])
        self. image_paths = image_paths

    def __getitem__(self, index):

        image   = self.images[index]
        caption = self.captions[index]
        path = self.image_paths[index]

        #t_image = self.transforms(image)
        return image, caption, path

    def __len__(self):  # return count of sample we have

        return len(self.images)