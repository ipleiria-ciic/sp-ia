# Miscellaneous
import time
import random
import numpy as np

# Utils
import Utils

# Torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# Optuna
import optuna
from optuna.trial import TrialState
from optuna.storages import RetryFailedTrialCallback

def get_dataset(dataroot):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = dset.ImageFolder(root=dataroot, transform=transform)

    return dataset

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, gpu_number):
        super(Generator, self).__init__()
        self.gpu_number = gpu_number
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_size, feature_size_generator * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_size_generator * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_size_generator * 8, feature_size_generator * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size_generator * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_size_generator * 4, feature_size_generator * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size_generator * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_size_generator * 2, feature_size_generator, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size_generator),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_size_generator, number_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
    
def create_generator():
    netG = Generator(gpu_number).to(device)
    netG = nn.DataParallel(netG, list(range(gpu_number)))
    netG.apply(weights_init)
    return netG

class Discriminator(nn.Module):
    def __init__(self, gpu_number):
        super(Discriminator, self).__init__()
        self.gpu_number = gpu_number
        self.main = nn.Sequential(
            nn.Conv2d(number_channels, feature_size_discriminator, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_size_discriminator, feature_size_discriminator * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size_discriminator * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_size_discriminator * 2, feature_size_discriminator * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size_discriminator * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_size_discriminator * 4, feature_size_discriminator * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_size_discriminator * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_size_discriminator * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    
def create_discriminator():
    netD = Discriminator(gpu_number).to(device)
    netD = nn.DataParallel(netD, list(range(gpu_number)))
    netD.apply(weights_init)
    return netD

def train(netG, netD, optimizerG, optimizerD, dataloader, criterion, fixed_noise, epoch):
    img_list = []
    g_losses = []
    d_losses = []
    iters = 0

    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, latent_size, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, n_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        g_losses.append(errG.item())
        d_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == n_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    return g_losses, d_losses, img_list

def evaluate(netG, netD, dataloader, criterion):
    netG.eval()
    netD.eval()
    g_losses = []
    d_losses = []
    
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)

            noise = torch.randn(b_size, latent_size, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake).view(-1)
            errD_fake = criterion(output, label)

            errD = errD_real + errD_fake
            d_losses.append(errD.item())

            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            g_losses.append(errG.item())
            
    return np.mean(g_losses), np.mean(d_losses)

def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 16, 64, log=True)
    beta1 = trial.suggest_float('beta1', 0.5, 0.9)
    patience = trial.suggest_int('patience', 5, 15, log=True)

    dataset = get_dataset(dataroot='../Datasets/TRM-UAP')

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [1000, len(dataset) - 1000])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    netG = create_generator()
    netD = create_discriminator()

    optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, latent_size, 1, 1, device=device)

    criterion = nn.BCELoss()

    min_lr = learning_rate / 10

    schedulerD = optim.lr_scheduler.ReduceLROnPlateau(optimizerD, mode='min', factor=0.2, patience=patience, min_lr=min_lr, verbose=True)
    schedulerG = optim.lr_scheduler.ReduceLROnPlateau(optimizerG, mode='min', factor=0.2, patience=patience, min_lr=min_lr, verbose=True)

    early_stopping = Utils.EarlyStopping(patience=20, verbose=True)

    for epoch in range(n_epochs):
        netG.train()
        netD.train()
        train(netG, netD, optimizerG, optimizerD, train_loader, criterion, fixed_noise, epoch)
        g_loss, d_loss = evaluate(netG, netD, val_loader, criterion)

        schedulerD.step(g_loss + d_loss)
        schedulerG.step(g_loss + d_loss)

        early_stopping(g_loss + d_loss, netG)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    return g_loss + d_loss

if __name__ == '__main__':
    start_time = time.time()

    # --- GPU Usage -----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Parameters ----------------------------------------------------------
    n_epochs = 200
    number_of_trials = 50
    random_seed = 1
    latent_size = 100
    number_channels = 3
    feature_size_generator = 64
    feature_size_discriminator = 64
    gpu_number = 1
    real_label = 1
    fake_label = 0

    # --- Random Seed ----------------------------------------------------------
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.use_deterministic_algorithms(True)

    # --- Optuna Studying ------------------------------------------------------
    study = optuna.create_study(
        study_name="pytorch_checkpoint",
        direction="minimize",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=number_of_trials)

    print(f"[INFO] GAN Trained. Time taken: {time.time() - start_time:.2f} seconds")

    # --- Results -------------------------------------------------------------
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("\nStudy statistics: ")
    print("Number of finished trials: ", len(study.trials))
    print("Number of pruned trials: ", len(pruned_trials))
    print("Number of complete trials: ", len(complete_trials))

    trial = study.best_trial
    print("Best trial:")
    print("Value: ", trial.value)
    print("Params: ")

    for key, value in trial.params.items():
        print("{}: {}".format(key, value))

    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)
    df = df.loc[df['state'] == 'COMPLETE']
    df = df.drop('state', axis=1)
    df = df.sort_values('value')
    df.to_csv('Optuna_Results.csv', index=False)
    print("\nOverall Results (ordered by accuracy):\n {}".format(df))

    # most_important_parameters = optuna.importance.get_param_importances(study, target=None)

    # print('\nMost important hyperparameters:')
    # for key, value in most_important_parameters.items():
    #     print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))