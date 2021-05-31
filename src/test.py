import torch
import torch.nn
import os
from model.gan_model import Generator, Discriminator
from utils_ import *


def main():

    latent_dim = 128
    gen_type = 'random_gen'

    #hair = ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 'pink', 'blue', 'black', 'brown', 'blonde']
    #eyes = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 'brown', 'red', 'blue']

    hair = 'blonde'
    eyes = 'blue'

    sample_dir = '../results/generated'
    gen_model_dir = '../results/checkpoints/model-[64]-[3]/G_1922.ckpt'

    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    hair_classes = len(hair_mapping)
    eye_classes = len(eye_mapping)

    device = 'cpu'



    G = Generator(latent_dim, hair_classes + eye_classes)
    prev_state = torch.load(gen_model_dir, map_location='cpu')
    G.load_state_dict(prev_state['model'])
    G = G.eval()

    if gen_type == 'fix_hair_eye':
        generate_with_attributes(G, device, latent_dim, hair_classes,
                               eye_classes, sample_dir, hair_color=hair, eye_color=eyes)
    elif gen_type == 'random_gen':
        generate_random(G, device, latent_dim, hair_classes, eye_classes, sample_dir)

if __name__ == "__main__":
    main()
