import torch
import torch.nn as nn
from datasets.PersonDataset import PersonDataset
from datasets.UnifiedDataset import UnifiedDataset

def generate_dataset(dataset_name, cfg, period, aug=False):

	if dataset_name == 'Person' or dataset_name == 'person':
		return PersonDataset('person', cfg, period, aug)
	elif dataset_name == 'Unified' or dataset_name == 'unified':
		return UnifiedDataset('Unified', cfg, period, aug)
	else:
		raise ValueError('generateData.py: dataset %s is not support yet'%dataset_name)
