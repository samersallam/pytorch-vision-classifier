import copy
import os
import time
import pandas as pd
import numpy as np
import pickle as pk
from tqdm import tqdm as tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import Counter
from sklearn.model_selection import train_test_split
from .utilities import ElapsedTime

class PytorchDatasetPreparation:
    
    def __init__(self, dataset_dir = '', splitting_parameters = {},
                       loading_parameters = {}, data_transforms = {}, 
                       show_images_dims_summary = False, use_stratify = True):
        """ Initialization """
        
        # Dataset directory
        self.show_images_dims_summary = show_images_dims_summary
        
        self.use_stratify = use_stratify
        
        if type(dataset_dir) is str:
            # Single dataset
            self.root_dir = os.path.join(os.getcwd(), dataset_dir)
            self.dataset_folder = NewImageFolder(self.root_dir)
        elif type(dataset_dir) is list:
            # Multiple datasets
            self.root_dir = [os.path.join(os.getcwd(), dr) for dr in dataset_dir]
            img_fld = NewImageFolder(self.root_dir[0])

            for dr in self.root_dir[1:]:
                tmp_folder = ImageFolder(dr)
                img_fld.samples.extend(tmp_folder.samples)

            img_fld.imgs = img_fld.samples
            img_fld.root = ''
            self.dataset_folder = img_fld
        
        self.splitting_parameters = splitting_parameters
        self.loading_parameters = loading_parameters
        self.data_transforms = data_transforms
        
        # Call details functions
        self.classes_details()
        self.samples_details = self.dataset_folder.samples_detalis()
        
        # Splitting
        self.train_validation_split()
        
    def classes_details(self): 
        # Extract classes details
        self.number_of_classes = len(self.dataset_folder.classes)
        self.classes_names = self.dataset_folder.classes
        self.classes_idx = list(self.dataset_folder.class_to_idx.values())
        self.classes_names_and_idx = self.dataset_folder.class_to_idx
        
    def train_validation_split(self):
        
        # Split the samples into training and validation samples
        labels = [s[1] for s in self.dataset_folder.samples]
        st = None
        if self.use_stratify:
            st = labels
            
        train_imgs, validation_imgs, _, _ = train_test_split(
            self.dataset_folder.samples, 
            labels,
            test_size = self.splitting_parameters['validation_ratio'], 
            random_state = self.splitting_parameters['splitting_random_state'],
            stratify=st)
        
        print('Done')
        # Create training dataset
        self.training_dataset = copy.deepcopy(self.dataset_folder)
        self.training_dataset.samples = train_imgs
        self.training_dataset.imgs = train_imgs
        self.training_dataset.transform = self.data_transforms['train']
        self.training_dataset_details = self.training_dataset.samples_detalis()
        
        # Create validation dataset
        self.validation_dataset = copy.deepcopy(self.dataset_folder)
        self.validation_dataset.samples = validation_imgs
        self.validation_dataset.imgs = validation_imgs
        self.validation_dataset.transform = self.data_transforms['validation']
        self.validation_dataset_details = self.validation_dataset.samples_detalis()
        
        # Create data loaders
        if self.loading_parameters['training_sampler_class'] is None:
            # No sampler is provided
            sampler = None
        else:
            # Extract the class and its parameters
            SamplerClass = self.loading_parameters['training_sampler_class']
            sampler_parameters = self.loading_parameters['training_sampler_parameters']
            
            # Check if it needs the data source as an argument
            data_source = sampler_parameters['data_source']
            del sampler_parameters['data_source']
            
            # Define the sampler object
            if data_source:
                sampler = SamplerClass(self.training_dataset, **sampler_parameters)
            else:
                sampler = SamplerClass(**sampler_parameters)
        
        # Define training dataset loader
        self.training_loader = DataLoader(self.training_dataset,
                                          sampler = sampler,
                                          batch_size = self.loading_parameters['training_batch_size'], 
                                          shuffle = self.loading_parameters['training_shuffle'], 
                                          num_workers = self.loading_parameters['training_num_workers'],
                                          pin_memory = self.loading_parameters['training_pin_memory'])
        
        # Define validation dataset loader
        self.validation_loader = DataLoader(self.validation_dataset, 
                                            batch_size = self.loading_parameters['validation_batch_size'], 
                                            shuffle = self.loading_parameters['validation_shuffle'], 
                                            num_workers = self.loading_parameters['validation_num_workers'],
                                            pin_memory = self.loading_parameters['validation_pin_memory'])
    
    def show_random_images_sample(self, sample_size = 4):
        """ This function is to show a set of images before transformations """
        if sample_size > self.samples_details['number_of_samples']:
            print('Sample size should be smaller than the dataset size')
            return
        
        rand_sample = np.random.permutation(self.samples_details['number_of_samples'])[:sample_size]
        plt.figure
        for ind in rand_sample:
            sample = self.dataset_folder[ind]
            plt.imshow(sample[0])
            plt.title('Image class : ' + str(sample[1]))
            plt.show()
    
    def show_images_after_transform(self, sample_size = 4, rand_sample = None, 
                                    load_from = 'train'):
        """ This function is to show a set of images after transformations """
        data_set = None
        
        if load_from not in ['train', 'validation']:
            print('Wrong value for load_from parameter')
            return
        
        if load_from == 'train':
            if sample_size > self.training_dataset_details['number_of_samples']:
                print('Sample size should be smaller than the training dataset size')
                return
          
            data_set = self.training_dataset
            number_of_samples = self.training_dataset_details['number_of_samples']
            
        if load_from == 'validation':
            if sample_size > self.validation_dataset_details['number_of_samples']:
                print('Sample size should be smaller than the validation dataset size')
                return

            data_set = self.validation_dataset
            number_of_samples = self.validation_dataset_details['number_of_samples']
        
        if rand_sample is None:
            rand_sample = np.random.permutation(number_of_samples)[:sample_size]
        else:
            rand_sample = np.array(rand_sample)
        
        plt.figure
        for ind in rand_sample:
            sample = data_set[ind]
            sample, label = sample[0], sample[1]
            sample = sample.numpy().transpose((1, 2, 0))   # Transpose since tensor is [c, h, w]
            mean = np.array([0.485, 0.456, 0.406])         # Mean for pretrained models
            std = np.array([0.229, 0.224, 0.225])          # Std for pretrained models
            sample = std * sample + mean
            sample = np.clip(sample, 0, 1)
            plt.imshow(sample)
            plt.title(self.classes_names[label])
            plt.show()
    
    def get_data_loader(self, training_phase):
        if training_phase:
            return self.training_loader
        else:
            return self.validation_loader
    
    def warm_up_epoch(self):
        with ElapsedTime('Warm up epoch').cpu(with_gpu=False):
            for _ in tqdm(self.training_loader, 'Training'):
                pass
              
            for _ in tqdm(self.validation_loader, 'Validation'):
                pass
            
            print('')
    
    def __repr__(self):
        obj_str = '* Root directory : \n' + str(self.root_dir) + '\n\n'
        
        obj_str += '* Classes details : \n'
        obj_str += 'Number of classes : '  + str(self.number_of_classes) + '\n'
        obj_str += 'Available classes : '  + str(self.classes_names_and_idx) + '\n\n'
        
        if self.show_images_dims_summary:
            self.dims_summary = self.dataset_folder.check_images_dimensions()
            obj_str += '* Images dimensions details : \n'

            obj_str += 'Minimum width : ' + str(self.dims_summary['min_width']) + ' \n'
            obj_str += 'Average width : ' + str(self.dims_summary['avr_width']) + ' \n'
            obj_str += 'Maximum width : ' + str(self.dims_summary['max_width']) + ' \n\n'

            obj_str += 'Minimum height : ' + str(self.dims_summary['min_height']) + ' \n'
            obj_str += 'Average height : ' + str(self.dims_summary['avr_height']) + ' \n'
            obj_str += 'Maximum height : ' + str(self.dims_summary['max_height']) + ' \n'

            obj_str += '\n\n'
        
        def show_samples_details(details, dataset_name):
            obj_str = '* ' + dataset_name + '\n'
            obj_str += '** Extensions details : \n'
            obj_str += 'Available extensions : '  + str(details['images_extensions']) + '\n'
            obj_str += 'Number of samples per extension : \n' + str(details['number_of_samples_per_extension']) + '\n\n'

            obj_str += '** Samples details : \n'
            obj_str += 'Number of samples : '  + str(details['number_of_samples']) + '\n'
            obj_str += 'Number of samples per class and per extension : \n'  + str(details['number_of_samples_per_class']) + '\n'
            obj_str += 'Classes percentages : '  + str(details['classes_percentages']) + '\n'
            obj_str += 'Classes weights : '  + str(details['classes_weights']) + '\n'
            obj_str += '\n\n'
            return obj_str
        
        obj_str += show_samples_details(self.samples_details, 'Dataset')
        obj_str += show_samples_details(self.training_dataset_details, 'Training Dataset')
        obj_str += show_samples_details(self.validation_dataset_details, 'Validation Dataset')
        return obj_str

# Monkey patching (Link this function with the class ImageFolder)
class NewImageFolder(ImageFolder):
    def __init__(self, root_dir):
        super(NewImageFolder, self).__init__(root_dir)
        
    def samples_detalis(self):
        # Count number of samples
        number_of_samples = self.samples.__len__()

        # Extract files extensions
        extensions = [os.path.splitext(pth[0])[-1] for pth in self.samples]

        # Replace any '.jpeg' by '.jpg'
        extensions = [ex if ex != '.jpeg' and ex != '.JPG' else '.jpg' for ex in extensions]

        # Count number of samples for each extensions
        images_extensions = list(set(extensions))
        number_of_samples_per_extension = pd.Series(Counter(extensions))

        # Create a dataframe for samples details
        columns = ['class_name', 'class_index', 'number_of_samples'] + images_extensions
        number_of_samples_per_class = pd.DataFrame(columns=columns)

        # Count number of samples for each class for each extension

        for cls_name in self.class_to_idx:
            # Extrac a class samples
            cls_ind = self.class_to_idx[cls_name]

            cls_samples = [s[0] for s in self.samples if s[-1] == cls_ind]

            # Count number of samples per extension per class
            number_of_samples_per_ext = []
            for ex in images_extensions:
                number_of_samples_per_ext.append(len(
                        [s for s in cls_samples if s.lower().endswith(ex)
                            or (s.lower().endswith('.jpeg') 
                            and ex == '.jpg')]))

            number_of_samples_per_class.loc[cls_ind] = \
            [cls_name , cls_ind, len(cls_samples), *number_of_samples_per_ext]

        number_of_samples_per_class.set_index('class_index', inplace=True)

        # Calculate the classes weights and percentages
        num_of_smpls_cls = number_of_samples_per_class['number_of_samples'].values
        classes_percentage = np.array((num_of_smpls_cls/num_of_smpls_cls.sum()),
                                      dtype=np.float32).round(3)

        # The higher the percentage. The lower the weight
        classes_weights = np.array((1/classes_percentage), dtype=np.float32).round(3)

        return {
            'number_of_samples': number_of_samples,
            'number_of_samples_per_class': number_of_samples_per_class,
            'classes_weights': classes_weights,
            'classes_percentages': classes_percentage,
            'images_extensions': images_extensions,
            'number_of_samples_per_extension': number_of_samples_per_extension
            }


    def check_images_dimensions(self):
        """ This function extracts images dimensions. If any file is corrupted, 
            its extenstion will be replaced by '.invalid' extension 
        """
        file_name = os.path.split(self.root)[-1] + '_imgs_dims_file.pkl'
        imgs_dims_file = os.path.join(os.getcwd() , file_name)

        try:
            # Read the dimensions from a file if it is available
            with open(imgs_dims_file, 'rb') as f:
                imgs_dims = pk.load(f)
        except:
            print('The dimensions file is not available. A new file will be created')
            imgs_dims = {}


        for i in tqdm(range(self.samples.__len__()), 
                      desc='Check the minimum and maximum image dimensions'):
            if self.samples[i][0] not in imgs_dims.keys():
                try:
                    # Read correctly
                    imgs_dims[self.samples[i][0]] = Image.open(self.samples[i][0]).size
                except:
                    # The file is corrupted (change the extension)
                    print('This file is corrupted : ', self.samples[i][0])
                    file_name, ext = os.path.splitext(self.samples[i][0])
                    os.rename(self.samples[i][0], file_name + '.invalid')

            # Write the dims each 500 iterations
            if i % 500 == 0:
                # Update the dims file
                with open(imgs_dims_file, 'wb') as f:
                    pk.dump(imgs_dims, f)

        # Update the dims file
        with open(imgs_dims_file, 'wb') as f:
            pk.dump(imgs_dims, f)

        # Compute the dimensions average
        imgs_dims = list(imgs_dims.values())
        average_dims = np.array(np.mean(np.array(imgs_dims), axis=0), dtype=np.int16)

        dims_summary = {}

        # Sort according to the height (number of rows)
        imgs_dims.sort(key = lambda x: x[1])
        dims_summary['min_height'] = imgs_dims[0][1]
        dims_summary['max_height'] = imgs_dims[-1][1]
        dims_summary['avr_height'] = average_dims[1]

        # Sort according to the width (number of columns)
        imgs_dims.sort(key = lambda x: x[0])
        dims_summary['min_width'] = imgs_dims[0][0]
        dims_summary['max_width'] = imgs_dims[-1][0]
        dims_summary['avr_width'] = average_dims[0]

        return dims_summary

    