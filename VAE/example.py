import torch
from models import VAE, VAE_cnn
from tools import load_data, compute_loss, test, vis_img, vis_lat
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

def config_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", type=int, default=0)
    parser.add_argument("--en_layers", type=int, default=3, help="num of encoder layers")
    parser.add_argument("--in_dim", type=int, default=28*28)
    parser.add_argument("--hid_dim", type=int, default=2000, help="hidden size")
    parser.add_argument("--lat_dim", type=int, default=100, help="latent space dimension")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--gen_num", type=int, default=100, help="the num of gen per dim(do not change)")
    parser.add_argument("--gen_s", type=float, default=-5.0, help="(do not change)")
    parser.add_argument("--gen_e", type=float, default=5.0, help="(do not change)")
    parser.add_argument("--expname", type=str, default='')
    parser.add_argument("--beta", type=float, default=1.0, help="for balancing rec loss and kl loss")
    parser.add_argument("--model_type", type=str, default='vae',
                        help="vae, vae-cnn, ...")
    return parser.parse_args()


args = config_parser()
# set cuda id
if args.use_cuda >= 0 and torch.cuda.is_available():
    args.device = torch.device(f"cuda:{args.use_cuda}")
else:
    args.device = torch.device("cpu")
print(f"We are using {args.device}")

# set save dir
if len(args.expname) == 0:
    writer = SummaryWriter()
else:
    writer = SummaryWriter(os.path.join('exp', args.expname))

for arg in vars(args):
    writer.add_text('args', f"{arg}: {getattr(args, arg)}")

# build model and optimizer
if args.model_type == 'vae':
    model = VAE(in_dim=args.in_dim,
                hid_dim=args.hid_dim,
                lat_dim=args.lat_dim,
                en_layers=args.en_layers).to(args.device)
elif args.model_type == 'vae-cnn':
    model = VAE_cnn(in_dim=args.in_dim,
                    hid_dim=args.hid_dim,
                    lat_dim=args.lat_dim,
                    en_layers=args.en_layers).to(args.device)
else:
    print("Unknown model!")
    exit(0)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# load data
train_loader, test_loader, vis_data = load_data(args.batch_size)

total_loss, total_loss_rec, total_loss_kl = [], [], []
i_interval = args.epoch // 10
beta = args.beta  # beta balances reconstruction loss and kl loss
for i in tqdm(range(args.epoch)):
    model.train()
    # args.beta = beta * min(4*(1+i) / args.epoch, 1.0)  # apply dynamic beta
    train_loss, rec_loss, kl_loss = 0, 0, 0
    for idx, (data, _) in enumerate(train_loader):
        data = data.to(args.device)
        y, mu, logvar2, _ = model(data)
        optimizer.zero_grad()
        loss, detail = compute_loss(data.view(-1, args.in_dim), y, mu, logvar2, args.beta)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        rec_loss += detail[0].item()
        kl_loss += detail[1].item()
        if (idx + 1) % args.i_print == 0:
            print(f"   Iter[{idx+1}] loss:{loss.item()}, rec loss:{detail[0].item()}, kl loss:{detail[1].item()}")
            
    total_loss.append(train_loss/len(train_loader))
    total_loss_rec.append(rec_loss/len(train_loader))
    total_loss_kl.append(kl_loss/len(train_loader))

    print(f">> Epoch[{i+1}]: loss: {total_loss[-1]}")
    writer.add_scalar('train loss', total_loss[-1], i)
    writer.add_scalar('train rec loss', total_loss_rec[-1], i)
    writer.add_scalar('train kl loss', total_loss_kl[-1], i)
    if (i+1) % i_interval == 0:
        vis_lat(i+1, model, args, writer)  # Plot sampled visualization
        test(i+1, test_loader, model, args, writer)  # Plot latent space visualization
        vis_img(i+1, vis_data.to(args.device), model, writer)  # Plot reconstructive visualization

writer.close()

