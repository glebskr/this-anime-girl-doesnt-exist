import torch
import numpy as np
import torchvision.utils as vutils
import random

hair_mapping = ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple',
                'pink', 'blue', 'black', 'brown', 'blonde']
hair_dict = {
    'orange': 0,
    'white': 1,
    'aqua': 2,
    'gray': 3,
    'green': 4,
    'red': 5,
    'purple': 6,
    'pink': 7,
    'blue': 8,
    'black': 9,
    'brown': 10,
    'blonde': 11
}

eye_mapping = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green',
               'brown', 'red', 'blue']
eye_dict = {
    'gray': 0,
    'black': 1,
    'orange': 2,
    'pink': 3,
    'yellow': 4,
    'aqua': 5,
    'purple': 6,
    'green': 7,
    'brown': 8,
    'red': 9,
    'blue': 10
}

def save_model(model, optimizer, step, file_path):

    state = {'model' : model.state_dict(),
             'optim' : optimizer.state_dict(),
             'step' : step}
    torch.save(state, file_path)
    return

def load_model(model, optimizer, file_path):

    prev_state = torch.load(file_path)

    model.load_state_dict(prev_state['model'])
    optimizer.load_state_dict(prev_state['optim'])
    start_step = prev_state['step']

    return model, optimizer, start_step


def get_random_label(batch_size, hair_classes, eye_classes):


    hair_code = torch.zeros(batch_size, hair_classes)
    eye_code = torch.zeros(batch_size, eye_classes)

    hair_type = np.random.choice(hair_classes, batch_size)
    eye_type = np.random.choice(eye_classes, batch_size)

    for i in range(batch_size):
        hair_code[i][hair_type[i]] = 1
        eye_code[i][eye_type[i]] = 1

    return torch.cat((hair_code, eye_code), dim = 1) # набор случайных меток классов

def generate_random(model, device, latent_dim, hair_classes, eye_classes,sample_dir):

    hair_tag = torch.zeros(64, hair_classes).to(device)
    eye_tag = torch.zeros(64, eye_classes).to(device)

    for i in range(64):
        hair_class = random.choice(list(hair_dict.values()))
        eye_class = random.choice(list(eye_dict.values()))
        hair_tag[i][hair_class], eye_tag[i][eye_class] = 1, 1

    tag = torch.cat((hair_tag, eye_tag), 1)
    z = torch.randn(64, latent_dim).to(device)

    output = model(z, tag)
    image_path = '{}/{} hair {} eyes.png'.format(sample_dir, 'random', 'random')
    vutils.save_image(output, image_path)
    return image_path


def generate_with_attributes(model, device, latent_dim, hair_classes, eye_classes,
                           sample_dir, hair_color='blonde', eye_color='blue'):

    hair_tag = torch.zeros(64, hair_classes).to(device)
    eye_tag = torch.zeros(64, eye_classes).to(device)


    hair_class = hair_dict[hair_color]
    eye_class = eye_dict[eye_color]
    for i in range(64):
        hair_tag[i][hair_class], eye_tag[i][eye_class] = 1, 1

    tag = torch.cat((hair_tag, eye_tag), 1)
    z = torch.randn(64, latent_dim).to(device)

    output = model(z, tag)
    image_path = '{}/{} hair {} eyes.png'.format(sample_dir, hair_mapping[hair_class], eye_mapping[eye_class])
    vutils.save_image(output, image_path)
    return image_path

