import torch
import torch.nn
import torch.optim as optim
import torchvision.utils as vutils
#import kornia.augmentation as K
import torch.nn as nn
import os, tqdm, re, glob
from datasets import train_loader
from model.gan_model import Generator, Discriminator
from utils_ import *



hair = ['orange', 'white', 'aqua', 'gray', 'green', 'red', 'purple', 'pink', 'blue', 'black', 'brown', 'blonde']
eyes = ['gray', 'black', 'orange', 'pink', 'yellow', 'aqua', 'purple', 'green', 'brown', 'red', 'blue']


def main():

    latent_dim = 128
    batch_size = 64
    iterations = 1
    checkpoint_dir = '../results/checkpoints'
    sample_dir = '../results/samples'
    lr = 0.0001
    sample_step = 70
    beta = 0.5
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    hair_classes, eye_classes = len(hair), len(eyes)
    num_classes = hair_classes + eye_classes

    smooth = 0.9

    config = 'model-[{}]-[{}]'.format(batch_size, iterations)
    print('Configuration: {}'.format(config))

    random_sample_dir = '{}/{}/random_generation'.format(sample_dir, config)
    fixed_attribute_dir = '{}/{}/fixed_attributes'.format(sample_dir, config)
    checkpoint_dir = '{}/{}'.format(checkpoint_dir, config)

    if not os.path.exists(random_sample_dir):
    	os.makedirs(random_sample_dir)
    if not os.path.exists(fixed_attribute_dir):
    	os.makedirs(fixed_attribute_dir)
    if not os.path.exists(checkpoint_dir):
    	os.makedirs(checkpoint_dir)

    G = Generator(latent_dim = latent_dim, class_dim = num_classes).to(device)
    D = Discriminator(hair_classes = hair_classes, eye_classes = eye_classes).to(device)

    G_optim = optim.Adam(G.parameters(), betas = [beta, 0.999], lr = lr)
    D_optim = optim.Adam(D.parameters(), betas = [beta, 0.999], lr = lr)

    start_step = 0
    models = glob.glob(os.path.join(checkpoint_dir,'/*.ckpt'))
    max_n = -1
    for model in models:
        n = int(re.findall(r'\d+', model)[-1])
        max_n = max(max_n, n)

    if max_n != -1:
        G, G_optim, start_step = load_model(G, G_optim, os.path.join(
            checkpoint_dir, 'G_{}.ckpt'.format(max_n)))
        D, D_optim, start_step = load_model(D, D_optim, os.path.join(
            checkpoint_dir, 'D_{}.ckpt'.format(max_n)))
        print("epoch start: ", start_step)

    criterion = torch.nn.BCELoss()
    #transform = nn.Sequential(
     #   K.RandomAffine(360),
    ##)

    ########## Начало обучения ##########
    for epoch in tqdm.trange(iterations, desc='Epoch Loop'):
        if epoch < start_step:
            continue
        print('Epoch :', epoch)
        for step_i, (real_img, hair_tags, eye_tags) in enumerate(tqdm.tqdm(train_loader, desc='Inner Epoch Loop')):
            real_label = torch.ones(batch_size).to(device)
            fake_label = torch.zeros(batch_size).to(device)
            soft_label = torch.Tensor(batch_size).uniform_(smooth, 1).to(device)
            real_img, hair_tags, eye_tags = real_img.to(
                device), hair_tags.to(device), eye_tags.to(device)


            for _ in range(1):
                # обучение дискриминатора
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_tag = get_random_label(batch_size = batch_size,
                                            hair_classes = hair_classes,
                                            eye_classes = eye_classes).to(device)

                fake_img = G(z, fake_tag).to(device)

                # aug_fake_img =  transform(fake_img)
                # aug_real_img = transform(real_img)

                d_real_img, d_fake_img = real_img, fake_img
                if epoch % 10 and epoch != 0 == 0:
                    d_real_img, d_fake_img = fake_img, real_img

                real_score, real_hair_predict, real_eye_predict = D(d_real_img)
                fake_score, _, _ = D(d_fake_img)

                real_discrim_loss = criterion(real_score, soft_label)
                fake_discrim_loss = criterion(fake_score, fake_label)

                real_hair_aux_loss = criterion(real_hair_predict, hair_tags)
                real_eye_aux_loss = criterion(real_eye_predict, eye_tags)
                real_classifier_loss = real_hair_aux_loss + real_eye_aux_loss

                discrim_loss = (real_discrim_loss + fake_discrim_loss) * 0.5

                D_loss = discrim_loss + real_classifier_loss
                D_optim.zero_grad()
                D_loss.backward()
                D_optim.step()

            if step_i % 100 == 0:
            	print('Discriminator loss: ' , D_loss.item())

            # обучение генератора
            for _ in range(1):
                z = torch.randn(batch_size, latent_dim).to(device)
                fake_tag = get_random_label(batch_size = batch_size,
                                            hair_classes = hair_classes,
                                            eye_classes = eye_classes).to(device)

                hair_tag = fake_tag[:, 0 : hair_classes]
                eye_tag = fake_tag[:, hair_classes : ]
                fake_img = G(z, fake_tag).to(device)

                fake_score, hair_predict, eye_predict = D(fake_img)
                discrim_loss = criterion(fake_score, real_label)
                hair_aux_loss = criterion(hair_predict, hair_tag)
                eye_aux_loss = criterion(eye_predict, eye_tag)
                classifier_loss = hair_aux_loss + eye_aux_loss

                G_loss = (classifier_loss + discrim_loss)
                G_optim.zero_grad()
                G_loss.backward()
                G_optim.step()
            if step_i % 100 == 0:
            	print('Generator loss: ' , G_loss.item())

            ########## чекпоинты ##########
            if epoch == 0 and step_i == 0:
                vutils.save_image(real_img, os.path.join(random_sample_dir, 'real.png'))

            if step_i % sample_step == 0:
                vutils.save_image(fake_img.data.view(batch_size, 3, 64, 64),
                                  os.path.join(random_sample_dir, 'fake_step_{}_{}.png'.format(epoch, step_i)))
            if step_i == 0:
                save_model(model=G, optimizer=G_optim, step=epoch,
                           file_path=os.path.join(checkpoint_dir, 'G_{}.ckpt'.format(epoch)))
                save_model(model=D, optimizer=D_optim, step=epoch,
                        file_path = os.path.join(checkpoint_dir, 'D_{}.ckpt'.format(epoch)))


                generate_with_attributes(model=G, device=device, latent_dim=latent_dim,
                                        hair_classes = hair_classes, eye_classes = eye_classes,
                                        sample_dir = fixed_attribute_dir)

if __name__ == '__main__':
    main()
