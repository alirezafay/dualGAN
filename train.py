import os
import torch
import argparse
from loss import *
import torch.nn as nn
from tqdm import tqdm
from data import *
from model import *

parser=argparse.ArgumentParser(description='train')

parser.add_argument('--epoch',type=int,default=24,help='config file')
parser.add_argument('--bs',type=int,default=2,help='config file')
parser.add_argument('--lr',type=float,default=0.002,help='config file')

args=parser.parse_args()

def train_Generator(model,l_min,loss_G,loss_adv,loss_dv,loss_di,vis,ir,max_epoch,lr):
	for i in model.G.parameters():
		i.requires_grad=True
	for i in model.Dv.parameters():
		i.requires_grad=False
	for i in model.Di.parameters():
		i.requires_grad=False
	opt=torch.optim.RMSprop(model.parameters(),lr)
	for epoch in range(0,max_epoch+1):
		fusion_v,fusion_i,score_v,score_i,score_Gv,score_Gi=model(vis,ir)
		loss_v=loss_dv(score_v,score_Gv)
		loss_i=loss_di(score_i,score_Gi)
		loss_adv_G=loss_adv(score_Gv,score_Gi)
		loss_g=loss_G(vis,ir,fusion_v,score_Gv,score_Gi)
		# print('G epoch:',epoch,'\tloss_adv:',loss_adv_G.item(),'\tloss_g:',loss_g.item())
		# if loss_v<l_min or loss_i<l_min:
		# 	opt.zero_grad()
		# 	loss_adv_G.backward()
		# 	opt.step()
		# elif loss_g>l_Gmax:
		# 	opt.zero_grad()
		# 	loss_g.backward()
		# 	opt.step()
		opt.zero_grad()
		loss_g.backward()
		opt.step()


	return model,loss_g

def train_Discriminators(model,l_max,loss_G,loss_dv,loss_di,vis,ir,max_epoch,lr):
	for i in model.G.parameters():
		i.requires_grad=False
	for i in model.Dv.parameters():
		i.requires_grad=True
	for i in model.Di.parameters():
		i.requires_grad=False
	opt=torch.optim.SGD(model.parameters(),lr)
	for epoch in range(0,max_epoch+1):
		_,_,score_v,_,score_Gv,_=model(vis,ir)
		loss_v=loss_dv(score_v,score_Gv)
		# print('Dv epoch:',epoch,score_v.item(),score_Gv.item(),'\tloss_v:',loss_v.item())
		if loss_v<=l_max:
			break
		opt.zero_grad()
		loss_v.backward()
		opt.step()

	for i in model.G.parameters():
		i.requires_grad=False
	for i in model.Dv.parameters():
		i.requires_grad=False
	for i in model.Di.parameters():
		i.requires_grad=True
	opt=torch.optim.SGD(model.parameters(),lr)
	for epoch in range(0,max_epoch+1):
		_,_,_,score_i,_,score_Gi=model(vis,ir)
		loss_i=loss_di(score_i,score_Gi)
		# print('Di epoch:',epoch,score_i.item(),score_Gi.item(),'\tloss_i:',loss_i.item())
		if loss_i<=l_max:
			break
		opt.zero_grad()
		loss_i.backward()
		opt.step()
	fusion_v,fusion_i,score_v,score_i,score_Gv,score_Gi=model(vis,ir)
	L_G=loss_G(vis,ir,fusion_v,score_Gv,score_Gi)
	return model,L_G,loss_i,loss_v

from torch.utils.data import DataLoader, Dataset
class GetDataset(Dataset):
    def __init__(self, MRIFolder, PETFolder,transform=None):
        self.MRIFolder = MRIFolder
        self.PETFolder = PETFolder
        self.transform = transforms.Compose(
        [transforms.Resize((256, 256)),
         transforms.ToTensor(),
         transforms.RandomHorizontalFlip(p=0.5),
	 transforms.Normalize([0.5],[0.5])
         ])
        self.mri_files = sorted(os.listdir(MRIFolder))
        self.pet_files = sorted(os.listdir(PETFolder))
        
    def __getitem__(self, index):
        mri_path = os.path.join(self.MRIFolder, self.mri_files[index])
        pet_path = os.path.join(self.PETFolder, self.pet_files[index])
        mri_image = Image.open(mri_path).convert('L')
        pet_image = Image.open(pet_path).convert('L')

        if self.transform:
            mri_image = self.transform(mri_image)
            pet_image = self.transform(pet_image)
            
        #input = torch.cat((pet_image, mri_image), -3)
        return pet_image, mri_image
        
    def __len__(self):
        return len(self.mri_files)


def main():
        l_max=1.8
        l_min=1.2
        max_epoch=5
        batch_size=args.bs
	DATASET_mri = '/kaggle/working/images/MRI'
        DATSAET_pet = '/kaggle/working/images/PET'
        dataset_train = GetDataset(MRIFolder=DATASET_mri, PETFolder=DATSAET_pet, transform=None )
        loader = DataLoader(dataset_train,shuffle=False,batch_size=batch_size)
        model=DDcGAN(if_train=True).cuda()
        Loss_G=L_G()
        Loss_adv_G=L_adv_G()
        Loss_Dv=L_Dv()
        Loss_Di=L_Di()
        for epoch in range(0,args.epoch):
                print(f'epoch:{epoch}')
                loss_generator = 0
                loss_discriminator_i = 0
                loss_discriminator_v = 0
		for i , (pet, image) in enumerate(loader):
			model,loss_g=train_Generator(model,l_min,Loss_G,Loss_adv_G,Loss_Dv,Loss_Di,vis,ir,max_epoch,args.lr)
                        loss_generator += loss_g.item()
                        model,loss_G,loss_i,loss_v=train_Discriminators(model,l_max,Loss_G,Loss_Dv,Loss_Di,vis,ir,max_epoch,args.lr)
                        loss_discriminator_i = loss_i.item()
                        loss_discriminator_v = loss_v.item()
                torch.save(model,'/kaggle/working'+str(epoch)+'.pth')
		print(f'Loss discriminator infrared: {loss_discriminator_i}')
		print(f'Loss discriminator visible: {loss_discriminator_v}')
		print(f'Loss Generator: {loss_generator}')

if __name__=='__main__':
	main()
