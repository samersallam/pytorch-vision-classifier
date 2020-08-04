import os
import numpy as np
from PIL import Image
import pickle as pk
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output
import tarfile

from .evaluation_metrics import ClassifierReport
from .pytorch_device_manager import  DeviceManager
from .pytorch_data_transformation import NCenterCrop

from .utilities import ElapsedTime
from .lr_finder import LRFinder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from copy import deepcopy
import mlflow


from classification_analysis.classification_analysis import ClassificationAnalysis

class ModelInitializer:
    def __init__(self, model_name, 
                 use_pretrained=True, 
                 update_head={'update': True,
                              'init_mode': 'xavier_normal',
                               'val': None},
                 num_classes=0, 
                 dropout = {'add_dropout': False, 'prob': None}):
        
        self.model_name = model_name
        self.model = None
        self.num_classes = num_classes
        self.update_head = update_head
        self.dropout = dropout
        self.use_pretrained = use_pretrained
    
    @staticmethod
    def update_trainability(layers: list, trainable):
        for layer in layers:
            for params in layer.parameters():
                params.requires_grad = trainable
    
    @staticmethod
    def init_layer_weight(layer, init_mode, val):
        
        if init_mode is None:
            return
          
        elif init_mode == 'uniform':
            torch.nn.init.uniform_(layer.weight)
            
        elif init_mode == 'normal':
            torch.nn.init.normal_(layer.weight)

        elif init_mode == 'constant':
            torch.nn.init.constant_(layer.weight, val)
            
        elif init_mode == 'ones':
            torch.nn.init.ones_(layer.weight)
            
        elif init_mode == 'zeros':
            torch.nn.init.zeros_(layer.weight)
            
        elif init_mode == 'eye':
            torch.nn.init.eye_(layer.weight)
                        
        elif init_mode == 'dirac':
            torch.nn.init.dirac_(layer.weight)
                        
        elif init_mode == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(layer.weight)
            
        elif init_mode == 'xavier_normal':
            torch.nn.init.xavier_normal_(layer.weight)
            
        elif init_mode == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(layer.weight)
            
        elif init_mode == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(layer.weight)
            
        elif init_mode == 'orthogonal':
            torch.nn.init.orthogonal_(layer.weight)
            
        elif init_mode == 'sparse':
            torch.nn.init.sparse_(layer.weight)
            
        if isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm3d):
            torch.nn.init.zeros_(layer.bias)
        
        elif layer.bias is not None:
            torch.nn.init.normal_(layer.bias)
            
    def creat_initialized_fc(self, num_ftrs):
        # Create a fc layer
        fc = nn.Linear(num_ftrs, self.num_classes)
        
        # Initialize the weights
        ModelInitializer.init_layer_weight(fc, self.update_head['init_mode'], self.update_head['val'])
        
        if self.dropout['add_dropout']:
            fc = nn.Sequential(nn.Dropout(self.dropout['prob']), fc)
        
        return fc
    
    def update_resnet_head(self):
        num_ftrs = self.model.fc.in_features
        
        self.model.fc = self.creat_initialized_fc(num_ftrs)
        
    def update_squeezenet_head(self):
        if self.dropout['add_dropout']:
            self.model.classifier[1] = nn.Sequential(nn.Dropout(self.dropout['add_dropout']),
                                                nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1)))
          
            ModelInitializer.init_layer_weight(self.model.classifier[1][1], self.update_head['init_mode'], self.update_head['val'])
          
        else:
            self.model.classifier[1] = nn.Conv2d(512, self.num_classes, kernel_size=(1,1), stride=(1,1))
            ModelInitializer.init_layer_weight(self.model.classifier[1], self.update_head['init_mode'], self.update_head['val'])
        
        self.model.num_classes = self.num_classes
        
    
    def update_densenet_head(self):
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = self.creat_initialized_fc(num_ftrs)
        
    def update_vgg_alexnet_head(self):
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = self.creat_initialized_fc(num_ftrs)
        
        
    def update_vgg_head(self):
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = self.creat_initialized_fc(num_ftrs)
        
    def update_inception_head(self):
        num_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = self.creat_initialized_fc(num_ftrs)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = self.creat_initialized_fc(num_ftrs)
    
    def get_model(self):
        if self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=self.use_pretrained)
            
        if self.model_name == 'resnet34':
            self.model = models.resnet34(pretrained=self.use_pretrained)
            
        if self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=self.use_pretrained)

        if self.model_name == 'resnet101':
            self.model = models.resnet101(pretrained=self.use_pretrained)
            
        if self.model_name == 'resnet152':
            self.model = models.resnet152(pretrained=self.use_pretrained)
            
        if self.model_name == 'squeezenet1_0':
            self.model = models.squeezenet1_0(pretrained=self.use_pretrained)
            
        if self.model_name == 'squeezenet1_1':
            self.model = models.squeezenet1_1(pretrained=self.use_pretrained)
            
        if self.model_name == 'densenet121':
            self.model = models.densenet121(pretrained=self.use_pretrained)
            
        if self.model_name == 'densenet169':
            self.model = models.densenet169(pretrained=self.use_pretrained)

        if self.model_name == 'densenet161':
            self.model = models.densenet161(pretrained=self.use_pretrained)  
            
        if self.model_name == 'densenet201':
            self.model = models.densenet201(pretrained=self.use_pretrained)  
            
        if self.model_name == 'alexnet':
            self.model = models.alexnet(pretrained=self.use_pretrained) 
            
        if self.model_name == 'vgg11':
            self.model = models.vgg11(pretrained=self.use_pretrained) 
            
        if self.model_name == 'vgg11_bn':
            self.model = models.vgg11_bn(pretrained=self.use_pretrained) 
            
        if self.model_name == 'vgg13':
            self.model = models.vgg13(pretrained=self.use_pretrained) 
            
        if self.model_name == 'vgg13_bn':
            self.model = models.vgg13_bn(pretrained=self.use_pretrained) 
            
        if self.model_name == 'vgg16':
            self.model = models.vgg16(pretrained=self.use_pretrained) 
            
        if self.model_name == 'vgg16_bn':
            self.model = models.vgg16_bn(pretrained=self.use_pretrained) 
            
        if self.model_name == 'vgg19':
            self.model = models.vgg19(pretrained=self.use_pretrained) 
            
        if self.model_name == 'vgg19_bn':
            self.model = models.vgg19_bn(pretrained=self.use_pretrained) 
            
        if self.model_name == 'inception_v3':
            self.model = models.inception_v3(pretrained=self.use_pretrained) 

        if self.update_head['update']:
            if 'resnet' in self.model_name:
                self.update_resnet_head()
               
            if 'squeezenet' in self.model_name:
                self.update_squeezenet_head()
            
            if 'densenet' in self.model_name:
                self.update_densenet_head()
                
            if 'vgg' in self.model_name:
                self.update_vgg_alexnet_head()
                
            if 'alexnet' in self.model_name:
                self.update_vgg_alexnet_head()
                
            if 'inception' in self.model_name:
                self.update_inception_head()
            
        ModelInitializer.update_trainability([self.model], trainable = False)
        
        return self.model

class ModelTraining:
  
    def __init__(self, model, model_name, device, loss_function = None, optimizer = None,
                       scheduler = None, num_epochs = None, 
                       input_type={'train': 'single_crop','validation': 'single_crop'},
                       save_model_rate = 5, save_last_model = True):
        
        # Initial
        if model != None:
            self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.plateau_scheduler = False
        self.num_epochs = num_epochs
        self.input_type = input_type
        self.phases = ['train', 'validation']
        self.inputs = self.labels = self.outputs = self.outputs_score = self.preds = self.loss = None
        self.current_epoch = 0
        
        # For model saving
        self.save_model_rate = save_model_rate   # [epoch]
        self.save_last_model = save_last_model
        
        # For model tracking
        self.last_run_id = None
        
        # For model steps timing
        self.steps_timing = dict()
        
        # For model evaluation
        self.all_epochs_data = list()
        self.best_metrics = dict()
        self.metrics = dict()
        _phase = ['val', 'train']
        _metrics = ['acc', 'kappa', 'recall', 'fscore', 'precision', 'spec', 'loss']        
        for ph in _phase:
            for met in _metrics:
                self.metrics[ph + '_' + met] = list()
                if met == 'loss':
                    self.best_metrics[ph + '_' + met] = float("inf")
                else:
                    self.best_metrics[ph + '_' + met] = 0
                   
    def data_loading(self, data_loader):
        self.inputs, self.labels = next(iter(data_loader))
        
    def data_transfer(self):
        self.inputs, self.labels = self.inputs.to(self.device), self.labels.to(self.device)
              
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def single_crop_forward(self):
        self.outputs = self.model(self.inputs)
    
    def multi_crops_forward(self):
        bs, ncrops, c, h, w = self.inputs.size()
        self.outputs = self.model(self.inputs.view(-1, c, h, w)).view(bs, ncrops, -1).mean(dim=1)
    
    def forward(self, input_type):
        
        if input_type == 'single_crop':
            self.single_crop_forward()
        
        elif input_type == 'multi_crops':
            self.multi_crops_forward()
    
    def get_predictions(self, dim=1):
        self.outputs_score = F.softmax(self.outputs, dim = dim)
        self.preds = self.outputs_score.argmax(dim = dim)
#         self.preds_prop = torch.cuda.FloatTensor(torch.tensor([self.outputs_score[i][self.preds[i]] for i in range(len(self.preds))],
#                                                               requires_grad=True).to(self.device))
    
    def loss_calculation(self):
        self.loss = self.loss_function(self.outputs, self.labels)
#         if isinstance(self.loss_function, nn.modules.loss._WeightedLoss):
#             self.loss = self.loss_function(self.outputs, self.labels)
          
#         else:
#             self.loss = self.loss_function(self.preds_prop, torch.cuda.FloatTensor(self.labels.float()))
    
    def backward(self):
        self.loss.backward()
        
    def optimizer_step(self):
        self.optimizer.step()
    
    def scheduler_step(self, best):
        if self.scheduler is not None:
            if self.plateau_scheduler:
                self.scheduler.step(self.best_metrics[best])
            else:
                self.scheduler.step()
    
    def set_model_state(self, training_phase):
        if training_phase:
            self.model.train()
        else:
            self.model.eval()
    
    def initialize_phase_data(self):
        # Empty lists to save the scores and predications
        self.y_loss, self.y_true, self.y_pred, self.y_score = [list() for _ in range(4)]
        
    def update_phase_data(self):
        # Store the results
        self.y_loss.append(self.loss.view(1))
        self.y_true.append(self.labels)
        self.y_pred.append(self.preds)
        self.y_score.append(self.outputs_score)
    
    def initialize_epoch_data(self):
        self.current_epoch += 1
        self.epoch_data = dict()
    
    def update_epoch_data(self, epoch, phase):
        self.epoch_data['current_epoch'] = epoch
        self.epoch_data['y_loss_' + phase] = torch.cat(self.y_loss).cpu().detach().numpy()
        self.epoch_data['y_true_' + phase] = torch.cat(self.y_true).cpu().detach().numpy()
        self.epoch_data['y_pred_' + phase] = torch.cat(self.y_pred).cpu().detach().numpy()
        self.epoch_data['y_score_'+ phase] = torch.cat(self.y_score).cpu().detach().numpy()
    
    def one_phase(self, phase, dataset):
        
        print('The current phase is {}'.format(phase))
        training_phase = phase == self.phases[0]
        
        self.set_model_state(training_phase)
        data_loader = dataset.get_data_loader(training_phase)
        
        self.initialize_phase_data()
        
        for self.inputs, self.labels in data_loader:
            self.data_transfer()
            self.zero_grad()
            
            with torch.set_grad_enabled(training_phase):
                self.forward(self.input_type[phase])
                self.get_predictions()
                self.loss_calculation()
            
            if training_phase:
                self.backward()
                self.optimizer_step()
                
            self.update_phase_data()
    
    def one_epoch(self, dataset, epoch, best, track):
        
        self.initialize_epoch_data()
        self.scheduler_step(best)
        
        for phase in self.phases:
            self.one_phase(phase, dataset)
            self.update_epoch_data(epoch, phase)

        self.all_epochs_data.append(self.epoch_data)

        # Clear the cell then display the result
        clear_output()
        print('Epoch {}/{}'.format(epoch+1, self.num_epochs))
        print('Current learning rate: {} '.format(self.optimizer.param_groups[0]['lr']))
        self.evaluation_metrics_calculation(dataset)
        self.evaluation_metrics_visualization()
        self.model_save(best, track)

    def evaluation_metrics_calculation(self, dataset):
        # Calculate the loss
        epoch_data = self.epoch_data
        self.metrics['train_loss'].append(epoch_data['y_loss_train'].mean())
        self.metrics['val_loss'].append(epoch_data['y_loss_validation'].mean())
        
        # Calculate the evaluation metrics
        self.train_report = ClassifierReport(epoch_data['y_true_train'], epoch_data['y_pred_train'], 
                                        epoch_data['y_score_train'], number_of_classes = dataset.number_of_classes,
                                        classes_labels = dataset.classes_names)
        self.metrics['train_acc'].append(self.train_report.overall_accuracy)
        self.metrics['train_kappa'].append(self.train_report.overall_cohen_kappa)
        self.metrics['train_recall'].append(self.train_report.overall_recall)
        self.metrics['train_fscore'].append(self.train_report.overall_f1_score)
        self.metrics['train_precision'].append(self.train_report.overall_precision)
        self.metrics['train_spec'].append(self.train_report.overall_specificity)
        
        
        self.val_report = ClassifierReport(epoch_data['y_true_validation'], epoch_data['y_pred_validation'], 
                                      epoch_data['y_score_validation'], number_of_classes = dataset.number_of_classes,
                                      classes_labels = dataset.classes_names)
        self.metrics['val_acc'].append(self.val_report.overall_accuracy)
        self.metrics['val_kappa'].append(self.val_report.overall_cohen_kappa)
        self.metrics['val_recall'].append(self.val_report.overall_recall)
        self.metrics['val_fscore'].append(self.val_report.overall_f1_score)
        self.metrics['val_precision'].append(self.val_report.overall_precision)
        self.metrics['val_spec'].append(self.val_report.overall_specificity)
    
    def evaluation_metrics_visualization(self):
        
        print('Training')
        self.train_report.show_all()
        
        print(50 * ' - ')
        
        print('Validation')
        self.val_report.show_all()
       
        ClassificationAnalysis.line_plot(self.metrics['train_loss'],      self.metrics['val_loss'],      'Loss')
        ClassificationAnalysis.line_plot(self.metrics['train_acc'],       self.metrics['val_acc'],       'Accuracy')
        ClassificationAnalysis.line_plot(self.metrics['train_kappa'],     self.metrics['val_kappa'],     'Cohen Kappa')
        ClassificationAnalysis.line_plot(self.metrics['train_fscore'],    self.metrics['val_fscore'],    'F1-Score')
        ClassificationAnalysis.line_plot(self.metrics['train_precision'], self.metrics['val_precision'], 'Precision')
        ClassificationAnalysis.line_plot(self.metrics['train_recall'],    self.metrics['val_recall'],    'Recall')
        ClassificationAnalysis.line_plot(self.metrics['train_spec'],      self.metrics['val_spec'],      'Specificity')
      
    
    def model_save(self, best, track):
        def save_main_parameters(self):
            to_be_saved = dict()
            to_be_saved['epoch_data'] = self.epoch_data
            to_be_saved['steps_timing'] = self.steps_timing
            to_be_saved['train_report'] = self.train_report
            to_be_saved['val_report'] = self.val_report
            to_be_saved['metrics'] = self.metrics
            to_be_saved['best_metrics'] = self.best_metrics
            to_be_saved['model_name'] = self.model_name
            to_be_saved['device'] = self.device
            to_be_saved['loss_function'] = self.loss_function
            to_be_saved['current_epoch'] = self.current_epoch
            to_be_saved['num_epochs'] = self.num_epochs
            to_be_saved['input_type'] = self.input_type
            to_be_saved['last_run_id'] = self.last_run_id

            params = self.optimizer.param_groups
            lr = [params[i]['lr'] for i in reversed(range(len(params)))]
            to_be_saved['optimizer_lr'] = lr
            return to_be_saved
        
        # Update the best metric
        model_improved = False
        for key in self.best_metrics:
            if (key in ['train_loss', 'val_loss']) and (self.metrics[key][-1] < self.best_metrics[key]) or \
               (key not in ['train_loss', 'val_loss']) and (self.metrics[key][-1] > self.best_metrics[key]):
                
                if best == key:    
                    model_improved = True
                    
                
                try:
                    if (self.metrics[best][-1] == self.metrics[best][-2]) and \
                       (self.metrics['val_loss'][-1] < self.best_metrics['val_loss']):

                        model_improved = True

                except:
                    pass
                    
                self.best_metrics[key] = self.metrics[key][-1]
            
        # Save the model
        file_name = self.model_name + '.pkl'
        should_be_saved = False
        if model_improved:
            # Tracking Experiments 
            if track:
                if self.last_run_id:
                    mlflow.delete_run(self.last_run_id)
                    
                with mlflow.start_run(run_name=self.model_name) as run:
                    self.last_run_id = mlflow.active_run().info.run_id
                    
                    mlflow.log_param('epoch', self.current_epoch)
                    mlflow.log_param('loss_function', self.loss_function)
                    mlflow.log_param('scheduler', self.scheduler)
                        
                    for key in self.metrics:
                        mlflow.log_metric(key, self.metrics[key][-1])
            
            should_be_saved = True
            to_save = {
                           'last_model': self.model if self.save_last_model else None,
                           'last_params': save_main_parameters(self),
                           
                           'best_model': self.model,
                           'best_params': save_main_parameters(self)
                      }
        
        elif self.epoch_data['current_epoch'] % self.save_model_rate == 0:
            should_be_saved = True
            if os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    prev_model = pk.load(f)
            else:
                prev_model = None
            
            to_save = {
                           'last_model': self.model if self.save_last_model else None,
                           'last_params': save_main_parameters(self),
                           
                           'best_model': prev_model['best_model'] if prev_model is not None else None,
                           'best_params': prev_model['best_params'] if prev_model is not None else None
            }
        
        if should_be_saved:
            with open(file_name, 'wb') as f:
                pk.dump(to_save, f)
        
    def loop(self, dataset, best='val_acc', reset_epoch_cnt=True, track=True):
        
        if reset_epoch_cnt:
            # If we want continue training
            self.current_epoch = 0
        
        for epoch in range(self.num_epochs):
            with ElapsedTime('One epoch').cpu(with_gpu=False):
                self.one_epoch(dataset, epoch, best, track)
                    
    def features_extraction_details(self, n_crops, source_layers, phase, iteration):
        clear_output()
        print('Number of crops is {}'.format(n_crops))
        print('Number of sources {}'.format(len(source_layers)))
        print('Current phase is {}'.format(phase))
        print('Iteration {}'.format(iteration))
    
    
    def extract_features(self, dataset, new_ds_dir, features_layer = {
                                                                   'last_layer': True,
                                                                   'classifier': True,
                                                                   'softmax': True
                                                               }):
      
        # Get number of crops if found
        n_crops = 1
        for tr in dataset.data_transforms['train'].transforms:
            if isinstance(tr, NCenterCrop):
                n_crops = tr.n
                break
        
        # Get features sources
        layers = list()  
        for k in features_layer:
            if features_layer[k]:
                if k == 'softmax':
                    layers.append('mean_' + k)
                    layers.append(k + '_mean')
                    
                else:
                    layers.append(k)
                    
        # Generate dataset name 
        experiment_name = lambda model_name, features, n_crops :  model_name + '_ftrs_' + features + '_crops' + str(n_crops) + '.pkl'
        
        # dataset dict
        datasets = {
            layer: {
                'name': experiment_name(self.model_name.split('/')[-1], layer, n_crops),
                'train': {'features': list(), 'labels': list()},
                'validation': {'features': list(), 'labels': list()}
            } for layer in layers
        }
        
        # divide model into three steps
        model_steps = dict()
        
        # Step 1: get last layer outputs
        temp = deepcopy(self.model)
        if isinstance(self.model, models.resnet.ResNet):
            temp.fc = nn.Identity()
            # Step 2: get fc layer outputs 
            model_steps['classifier'] = self.model.fc
            
        elif isinstance(self.model, models.squeezenet.SqueezeNet):
            temp.classifier = nn.Identity()
            # Step 2: get fc layer outputs 
            model_steps['classifier'] = self.model.classifier
            
        elif isinstance(self.model, models.densenet.DenseNet):
            temp.classifier = nn.Identity()
            # Step 2: get fc layer outputs 
            model_steps['classifier'] = self.model.classifier
            
        model_steps['last_layer'] = temp
        
        
        
        # Step 3: get softmax outputs 
        model_steps['softmax'] = F.softmax

        for phase in self.phases:
            training_phase = phase == self.phases[0]
            
            # Set the model state to Evaluation
            self.set_model_state(False)
            
            # Get the dataset
            ds = dataset.get_data_loader(training_phase).dataset
            
            iteration = 0
            # Iterate over data
            for self.inputs, self.labels in ds:
                
                # Print some details
                if iteration % 1000 == 0: 
                    self.features_extraction_details(n_crops, layers, phase, iteration)
                
                iteration+= 1
                
                self.inputs = self.inputs.to(self.device)
                                
                with torch.no_grad():
                    
                    # Step 1: feed model with inputs and get last layer outputs 
                    last_layer_outputs = model_steps['last_layer'](self.inputs)
                    last_layer_outputs_mean = last_layer_outputs.mean(dim = 0)
                    
                    if features_layer['last_layer']:
                        datasets['last_layer'][phase]['features'].append(last_layer_outputs_mean.cpu().detach().numpy())
        
                    # Step 2: feed fc layer with last layer outputs 
                    classifier_outputs = model_steps['classifier'](last_layer_outputs)
                    classifier_outputs_mean = classifier_outputs.mean(dim = 0)
                    
                    if features_layer['classifier']:
                        datasets['classifier'][phase]['features'].append(classifier_outputs_mean.cpu().detach().numpy())
              
            
                    # Step 3: feed softmax with classifier outputs 
                    if features_layer['softmax']:
                        mean_softmax_outputs = model_steps['softmax'](classifier_outputs_mean, dim=0)
                        softmax_outputs_mean = model_steps['softmax'](classifier_outputs, dim = 1).mean(dim = 0)
                      
                        datasets['mean_softmax'][phase]['features'].append(mean_softmax_outputs.cpu().detach().numpy())
                        datasets['softmax_mean'][phase]['features'].append(softmax_outputs_mean.cpu().detach().numpy())                    
                    
                    # Add labels
                    for k in datasets:
                        datasets[k][phase]['labels'].append(np.array(self.labels))
                
                
        # Stack features and labels
        for phase in self.phases:
            for k in datasets:
                datasets[k][phase]['features'] = np.vstack(datasets[k][phase]['features'])
                datasets[k][phase]['labels'] = np.hstack(datasets[k][phase]['labels'])
                
        # Save datasets        
        for k in datasets:
            file_name = os.path.join(new_ds_dir, datasets[k]['name'])
                
            with open(file_name, 'wb') as f:
                pk.dump(datasets[k], f)

    
    def find_best_learning_rate(self, dataset, use_val_loss = False, end_lr = 10, 
                                num_iter=100, step_mode='exp'):
    
        # For more information refer to https://github.com/davidtvs/pytorch-lr-finder
        lr_finder = LRFinder(self.model, self.optimizer, self.loss_function, self.device)
    
        if use_val_loss:
            # Use validation loss as an indicator
            lr_finder.range_test(dataset.training_loader, val_loader=dataset.validation_loader, 
                           end_lr=end_lr, num_iter=num_iter, step_mode=step_mode)
      
        else:
            # Use training loss as an indicator
            lr_finder.range_test(dataset.training_loader, end_lr=end_lr, num_iter=num_iter, step_mode=step_mode)
    
        # Plot the curve
        lr_finder.plot()
    
        # Reset the model and optimizor to their original status
        lr_finder.reset()
    
    def model_memory_size(self, bits = 32):
        """ Calculate the model size in MB """
        total_bits = 0
        trainable_param = 0
        all_trainable_param = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_param += np.prod(np.array(param.shape))
                
            all_trainable_param += np.prod(np.array(param.shape))
            total_bits += np.prod(np.array(param.shape)) * bits
            
        
        model_memory_size = round(10**-6 * total_bits / 8)
        print('model size {} MB'.format(model_memory_size))
        print('number of all trainable parametesrs: {}'.format(all_trainable_param))
        print('number of current trainable parametesrs: {}'.format(trainable_param))
    
    def model_memory_utilization(self, input_type='single_crop', batch_size=64, crops=10, dim=(224, 224)):
        dvc_mng = DeviceManager()
       
        if input_type == 'single_crop':
            input_size = (batch_size, 3, dim[0], dim[1])
        elif input_type == 'multi_crops':
            input_size = (batch_size, crops, 3, dim[0], dim[1])
        
        self.inputs = torch.FloatTensor(np.ndarray(input_size))
        self.labels = torch.LongTensor(np.zeros(batch_size))
        
        with dvc_mng.get_last_gpu_usage('Data Transfer'):
            self.data_transfer()
        try:
            with dvc_mng.get_last_gpu_usage('Zero gradients'):
                 self.zero_grad()
        except:
            pass
          
        with dvc_mng.get_last_gpu_usage('Forward'):
            self.forward(input_type)

        with dvc_mng.get_last_gpu_usage('Prediction'):
            self.get_predictions()

        with dvc_mng.get_last_gpu_usage('Loss'):
            self.loss_calculation()

        with dvc_mng.get_last_gpu_usage('Backward'):
            self.backward()
            
        try:
            with dvc_mng.get_last_gpu_usage('Optimizor'):
                self.optimizer_step()
        except:
          pass
    
    def initialize_steps_timing(self):
        self.steps_timing = {
            'd_load': list(),
            'd_transfer': list(),
            'zero_grads': list(),
            'forward': list(),
            'prediction': list(),
            'loss': list(),
            'backward': list(),
            'optimizer': list()
        }
    
    def for_loop_timing(self, dataset, number_iters = 50, show_time=True):
        self.initialize_steps_timing()
        data_loader = dataset.get_data_loader(True)
        
        cnt = 0
        for self.inputs, self.labels in data_loader:
            self.common_operations_timing(show_time, self.input_type['train'])
            
            cnt += 1
            if cnt == number_iters:
                break
     
    def while_loop_timing(self, dataset, number_iters = 50, show_time=True):
        self.initialize_steps_timing()
        data_loader = dataset.get_data_loader(True)
        
        cnt = 0
        while cnt != number_iters:
            with ElapsedTime('Data loading', times_list=self.steps_timing['d_load'], show_time=show_time).cpu(with_gpu=True):
                self.data_loading(data_loader)
            
            self.common_operations_timing(show_time, self.input_type['train'])
            
            cnt += 1
            
    def common_operations_timing(self, show_time, input_type='single_crop'):
        with ElapsedTime('Data transfer', times_list=self.steps_timing['d_transfer'], show_time=show_time).cpu(with_gpu=True):
            self.data_transfer()
        
        with ElapsedTime('Zero gradients', times_list=self.steps_timing['zero_grads'], show_time=show_time).gpu():
            self.zero_grad()
        
        with ElapsedTime('Forward', times_list=self.steps_timing['forward'], show_time=show_time).gpu():
            self.forward(input_type)
            
        with ElapsedTime('Prediction', times_list=self.steps_timing['prediction'], show_time=show_time).gpu():
            self.get_predictions()
        
        with ElapsedTime('Loss', times_list=self.steps_timing['loss'], show_time=show_time).gpu():
            self.loss_calculation()
        
        with ElapsedTime('Backward', times_list=self.steps_timing['backward'], show_time=show_time).gpu():
            self.backward()
        
        with ElapsedTime('Optimizer', times_list=self.steps_timing['optimizer'], show_time=show_time).gpu():
            self.optimizer_step()
        
        if show_time:
            print(20 * '-')
    
    @staticmethod
    def get_fetures_dataset(features_dir, dataset_name):
        features_dir = os.path.join(features_dir, dataset_name)
        with open(features_dir, 'rb') as f:
            return pk.load(f)
    
    @staticmethod
    def restore_model_training(param_dict, model_):   
        
        model_training = ModelTraining(model = model_,
                         model_name = param_dict['model_name'],
                         device = param_dict['device'],
                         loss_function = param_dict['loss_function'],
                         optimizer = None,
                         scheduler = None,
                         num_epochs = param_dict['num_epochs'],
                         input_type = param_dict['input_type'])
                
        model_training.epoch_data = param_dict['epoch_data']
        model_training.steps_timing = param_dict['steps_timing']
        model_training.train_report = param_dict['train_report']
        model_training.val_report = param_dict['val_report']
        model_training.metrics = param_dict['metrics']
        model_training.best_metrics = param_dict['best_metrics']
        model_training.current_epoch = param_dict['current_epoch']
        model_training.last_run_id = param_dict['last_run_id']
        
        return model_training
      
    @staticmethod
    def restore_last_model_training(model_name):     
        with open(model_name, 'rb') as f:
            attr = pk.load(f)
        last = attr['last_params']
        
        if last == None:
          return last
        
        return ModelTraining.restore_model_training(last, attr['last_model'])
    
    @staticmethod
    def restore_best_model_training(model_name):  
        with open(model_name, 'rb') as f:
            attr = pk.load(f)
        
        best = attr['best_params']
        
        if best == None:
          return best
        
        return ModelTraining.restore_model_training(best, attr['best_model'])
    
    
    @staticmethod
    def best_model_metrics_visualization(best_model_name):
        model_training = ModelTraining.restore_best_model_training(best_model_name)
        model_training.evaluation_metrics_visualization()
        
        
    @staticmethod
    def last_model_metrics_visualization(model_name):
        model_training = ModelTraining.restore_last_model_training(model_name)
        model_training.evaluation_metrics_visualization()
    
    @staticmethod
    def pth_model_save(model_tr_path, model_name):
        """
        model_path: path of the model training file
        model_name: name of the .pth file
        """
        with open(model_tr_path, 'rb') as f:
            attr = pk.load(f)
        
        torch.save(attr['best_model'].cpu().state_dict(), model_name)
    
    @staticmethod
    def compress_model_file(model_name):
        """Useful for sagemaker deployment
           model_name: name of the .pth file
        """
        output_filename = model_name.split('.')[0] + '.tar.gz'
        
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(model_name, arcname=os.path.basename(model_name))
            
    def trainable_layers_names_list(self):
        layers = set([])
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layers.add(name.rstrip('.' + name.split('.')[-1]))
        
        return layers
            
    def display_misclassification(self, model_name, dataset, to_display=10):
        with open(model_name, 'rb') as f:
            model_training = pk.load(f)
            
        ClassificationAnalysis.misclassification(model_training, dataset, to_display)


