import torch
import torch.nn as nn

    
class Generator(nn.Module):



    def __init__(self, latent_dim, class_dim):

        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.class_dim = class_dim

        self.gen = nn.Sequential(
            nn.ConvTranspose2d(in_channels = self.latent_dim + self.class_dim, 
                                out_channels = 1024, 
                                kernel_size = 4,
                                stride = 1,
                                bias = False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = 1024,
                                out_channels = 512,
                                kernel_size = 4,
                                stride = 2,
                                padding = 1,
                                bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = 512,
                                out_channels = 256,
                                kernel_size = 4,
                                stride = 2,
                                padding = 1,
                                bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = 256,
                                out_channels = 128,
                                kernel_size = 4,
                                stride = 2,
                                padding = 1,
                                bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.ConvTranspose2d(in_channels = 128,
                                out_channels = 3,
                                kernel_size = 4,
                                stride = 2,
                                padding = 1),
            nn.Tanh()
        )
        return
    
    def forward(self, _input, _class):

        concat = torch.cat((_input, _class), dim = 1)
        concat = concat.unsqueeze(2).unsqueeze(3)
        return self.gen(concat)

class Discriminator(nn.Module):

    def __init__(self, hair_classes, eye_classes):
        super(Discriminator, self).__init__()

        self.hair_classes = hair_classes
        self.eye_classes = eye_classes
        
        self.conv_layers = nn.Sequential(
                    nn.Conv2d(in_channels = 3, 
                             out_channels = 128, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 128, 
                             out_channels = 256, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 256, 
                             out_channels = 512, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace = True),
                    nn.Conv2d(in_channels = 512, 
                             out_channels = 1024, 
                             kernel_size = 4,
                             stride = 2,
                             padding = 1,
                             bias = False),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2, inplace = True)
                    )   
        self.discriminator_layer = nn.Sequential(
                    nn.Conv2d(in_channels = 1024, 
                             out_channels = 1, 
                             kernel_size = 4,
                             stride = 1),
                    nn.Sigmoid()
                    ) 
        self.bottleneck = nn.Sequential(
                    nn.Conv2d(in_channels = 1024, 
                             out_channels = 1024, 
                             kernel_size = 4,
                             stride = 1),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2)
                    )
        self.hair_classifier = nn.Sequential(
                    nn.Linear(1024, 128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, self.hair_classes),
                    nn.Softmax()
                    )
        
        self.eye_classifier = nn.Sequential(
                    nn.Linear(1024, 128),
                    nn.BatchNorm1d(128),
                    nn.LeakyReLU(0.2),
                    nn.Linear(128, self.eye_classes),
                    nn.Softmax()
                    )

        return
    
    def forward(self, _input):


        features = self.conv_layers(_input)  
        discrim_output = self.discriminator_layer(features).view(-1)
        
        flatten = self.bottleneck(features).squeeze()
        hair_class = self.hair_classifier(flatten)
        eye_class = self.eye_classifier(flatten) 
        return discrim_output, hair_class, eye_class

if __name__ == '__main__':
    latent_dim = 128
    class_dim = 22
    batch_size = 2
    z = torch.randn(batch_size, latent_dim)
    c = torch.randn(batch_size, class_dim)
    
    G = Generator(latent_dim, class_dim)
    D = Discriminator(12, 10)
    o = G(z, c)
    print(o.shape)
    x, y, z = D(o)
    print(x.shape, y.shape, z.shape)