import torch
from torch.autograd import Variable

import os 

from models.all_models import all_models
from data.data_loaders import get_data

import matplotlib.pyplot as plt


class WGAN_trainer:
    def __init__(self, opts):
        self._options=opts

        model_server               = all_models(self._options)
        self.G                     = model_server.G
        self.D                     = model_server.D
        self.generate_latent_space = model_server.latent_space_generator

        # these could be parsed in the options, but ok 
        self.n_critic=self._options.n_critic
        self.batch_size=self._options.batch_size
        self.alpha=self._options.alpha
        self.c=self._options.clipping_value
        self.generator_iters=self._options.generator_iters



        # Get the data
        self.train_loader,_ = get_data(self._options).get_data_loader(self.batch_size)
        self.postProcessSamples=get_data(self._options).get_postProcessor()

        # Decide if we will (if we can) use the GPU
        self.cuda = self._options.cuda and torch.cuda.is_available()
        if not torch.cuda.is_available() and self._options.cuda:
            print("Warning, you tried to use cuda, but its not available. Will use the CPU")
        if self.cuda:
            self.cuda_index=self._options.cuda_index
        
            
        if self.cuda: 
            self.G.cuda(self.cuda_index)
            self.D.cuda(self.cuda_index)


    def get_torch_variable(self, arg ):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)
        
    def trainIt(self):

        # Two helpers
        def get_infinite_batches(data_loader):
            while True:
                for i, images in enumerate(data_loader):
                    yield images
        data = get_infinite_batches(self.train_loader)



        optim_discriminator = torch.optim.RMSprop( self.D.parameters(), lr=self.alpha)
        optim_generator     = torch.optim.RMSprop( self.G.parameters(), lr=self.alpha)

        values_g_loss_data     =[]
        values_d_loss_fake_data=[]
        values_d_loss_real_data=[]
        for g_iter in range(self.generator_iters):
                
            for p in self.D.parameters():
                p.requires_grad=True
            for t in range(self.n_critic):
                self.D.zero_grad() # we need to set the gradients to zero, otherwise pytorch will sum them every time we call backward
                images=data.__next__()
                if (images.size()[0] != self.batch_size): # the dataset may not be multiple of the batch size
                    continue
                real_data=self.get_torch_variable(images)
                fake_data=self.get_torch_variable(self.generate_latent_space(self.batch_size) ) # TODO: the latent space is hardcoded, should be an input (use a lambda function in the models.)
                loss_a=torch.mean(self.D(real_data)-self.D(self.G(fake_data)))
                loss_a.backward() # compute gradients 
                optim_discriminator.step() # move the parameters

                # clip the parameters
                for p in self.D.parameters():
                    p.data.clamp_(-self.c,self.c)


                # Get the components of the loss to store them 
                loss_a_real_data=torch.mean(self.D(real_data)).data.cpu()
                loss_a_fake_data=torch.mean(-self.D(self.G(fake_data))).data.cpu()

                print(f'  Discriminator iteration: {t}/{self.n_critic}, loss_fake: {loss_a_fake_data}, loss_real: {loss_a_real_data}')
            

            # to avoid computation
            for p in self.D.parameters():
                p.requires_grad = False  
            
            self.G.zero_grad()  # we need to set the gradients to zero, otherwise pytorch will sum them every time we call backward

            fake_data=self.get_torch_variable(self.generate_latent_space(self.batch_size))
            loss_b=torch.mean(self.D(self.G(fake_data))) # because the gradient then goes with a minus
            loss_b.backward()
            optim_generator.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {loss_b.data.cpu()}')

            # to plot 
            values_g_loss_data     .append(loss_b.data.cpu() )
            values_d_loss_fake_data.append(loss_a_fake_data   )
            values_d_loss_real_data.append(loss_a_real_data   )

            if not (g_iter%1000):
                self.save_model(label=f"gen_iter_{g_iter}")
                self.generate_samples(1000, label=f"gen_iter_{g_iter}", load_model=False)

        
        fig, ax = plt.subplots()
        plot1=ax.plot( range(len(values_g_loss_data     )),values_g_loss_data      , label='loss generator')
        plot2=ax.plot( range(len(values_d_loss_fake_data)),values_d_loss_fake_data , label='loss critic fake data')
        plot3=ax.plot( range(len(values_d_loss_real_data)),values_d_loss_real_data , label='loss critic real data')
        plt.legend(handles=[plot1[0],plot2[0],plot3[0]])
        plt.savefig('training_%s.png'%self._options.trainingLabel)
        ax.clear()
        plt.close()
        self.save_model()

    def save_model(self,label=""):
        torch.save(self.G.state_dict(), f'{self._options.trainingLabel}_generator_{label}.pkl')
        torch.save(self.D.state_dict(), f'{self._options.trainingLabel}_discriminator_{label}.pkl')
        print(f'Models save to {self._options.trainingLabel}_discriminator_{label}.pkl & {self._options.trainingLabel}_generator_{label}.pkl')

    def load_model(self):
        # usually postprocessing is done in the cpu, but could be customized in the future
        self.G.load_state_dict(torch.load(f'./{self._options.trainingLabel}_generator.pkl',map_location=torch.device('cpu')))
        self.D.load_state_dict(torch.load(f'./{self._options.trainingLabel}_discriminator.pkl',map_location=torch.device('cpu'))) 


    def generate_samples(self, number_of_samples, label="",load_model=True):
        if load_model:
            self.load_model()
        samples=[]
        for _ in range(number_of_samples):
            z=self.get_torch_variable(self.generate_latent_space(1) )
            sample=self.G(z).data.cpu()
            samples.append( sample ) 
            print(sample)
        self.postProcessSamples( samples, label ) 

if __name__=="__main__":

    from optparse import OptionParser
    parser = OptionParser()

    parser.add_option("-m", "--model",           dest="model", type="string", default="convNNforNist", help="Architecture of the generator and critic. It also fixes the latent space distribution");
    parser.add_option("--n_critic",           dest="n_critic", type="int", default=5, help="Number of iterations of the critic per generator iteration");
    parser.add_option("--generator_iters",           dest="generator_iters", type="int", default=40000, help="Number of generator iterations");
    parser.add_option("--batch_size",           dest="batch_size", type="int", default=64, help="Mini-batch size");
    parser.add_option("--alpha",           dest="alpha", type="float", default=0.00005, help="Learning rate");
    parser.add_option("--clipping_value",           dest="clipping_value", type="float", default=0.01, help="Clipping parameters of the discriminator between (-c,c)");
    parser.add_option("--data",           dest="data", type="string", default='mnist', help="Dataset to train with");
    parser.add_option("--no-cuda",           dest="cuda", action='store_false', default=True, help="Do not try to use cuda. Otherwise it will try to use cuda only if its available");
    parser.add_option("--cuda_index",           dest="cuda_index", type="int", default=0, help="Index of the device to use");
    parser.add_option("--do_what",           dest="do_what", action='append', type="string", default=[], help="What to do");
    parser.add_option("--trainingLabel",           dest="trainingLabel",  type="string", default='trainingv1', help="Label where store to/read from the models");
    (options, args) = parser.parse_args()

    model = WGAN_trainer(options)
    if 'train' in options.do_what:
        model.trainIt()

    if 'generate' in options.do_what:
        model.generate_samples(1000)
