import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class EMD(nn.Module):
    def __init__(self, distance_matrix, classes_weights = None):
        super(EMD, self).__init__()
        self.distance_matrix = distance_matrix
        self.classes_weights = classes_weights
    
    @staticmethod
    def generate_distance_matrix(num_of_dims, factor):
        distance_matrix = list()
        for i in range(num_of_dims):
            row = list()
            for j in range(num_of_dims):
                dif = i - j if i >= j else j - i
                row.append(dif * factor)
            distance_matrix.append(row)
        
        return torch.FloatTensor(distance_matrix)
    
    def forward(self, outputs, labels):
        probabilities = F.softmax(outputs, dim=1)
        dists = self.distance_matrix[labels]
        
        if self.classes_weights is None:
            return torch.sum(probabilities*dists, dim=1).mean()
        else:
            return torch.sum(probabilities*dists*self.classes_weights, dim=1).mean()
    
    def __repr__(self):
        return 'The distance matrix is : ' + '\n' + str(self.distance_matrix)

