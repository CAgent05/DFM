import argparse
import torch
from collections import deque
from utils import prepare_agent, transform_input
import numpy as np
from src.net import GNNStack
from src.utils import AverageMeter
import pandas as pd
import os
import warnings
from PIL import Image

warnings.filterwarnings("ignore")



top1 = AverageMeter('Acc', ':6.2f')

parser = argparse.ArgumentParser(description='DRL Analysis based on TodyNet')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BipedalWalkerHC')
parser.add_argument('-n', '--nsteps', type=int, default=20)
parser.add_argument('-e', '--episodes', type=int, default=1000)
parser.add_argument('--alg', type=str, default="Todynet", help='the algorithm used for training')
parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')
parser.add_argument('--groups', type=int, default=4, help='the number of time series groups (num_graphs)')
parser.add_argument('--pool_ratio', type=float, default=0.2, help='the ratio of pooling for nodes')
parser.add_argument('--kern_size', type=str, default="9,5,3", help='list of time conv kernel size for each layer')
parser.add_argument('--in_dim', type=int, default=64, help='input dimensions of GNN stacks')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimensions of GNN stacks')
parser.add_argument('--out_dim', type=int, default=256, help='output dimensions of GNN stacks')

args = parser.parse_args()

model_dir = './model/' + args.alg + '/' + args.dataset + '_' + str(args.nsteps) + '.pth'

result_save_dir = './result/' + args.alg 
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)

result_save_dir = result_save_dir + '/' + args.dataset + '_' + str(args.nsteps) + '.csv'


args.kern_size = [ int(l) for l in args.kern_size.split(",") ]


df = pd.DataFrame(columns=['Episode', 'Reward', 'Pre', 'True', 'Probabilities', 'Steps'])

if args.dataset[-2:] == "SA":
    input_tag = True
    args.dataset = args.dataset[:-2]
else:
    input_tag = False


env, model, num_nodes, alg_tag = prepare_agent(args.dataset, input_tag)
print(num_nodes, alg_tag)
    
seq_length = 20

todeynet = GNNStack(gnn_model_type=args.arch, num_layers=args.num_layers, 
                    groups=args.groups, pool_ratio=args.pool_ratio, kern_size=args.kern_size, 
                    in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim, 
                    seq_len=seq_length, num_nodes=num_nodes, num_classes=2)
todeynet.load_state_dict(torch.load(model_dir))  #r'/home/cy/WorkForISSRE/code/model/InvertedDoublePendulumACVA.pth'
todeynet.to('cuda:0')
todeynet.eval()

check_episode = args.episodes

pre_label = np.zeros(check_episode)
true_label = np.zeros(check_episode)
output_dir = 'RQ4/pic/' + args.dataset + '/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

state_save = []  
save_cnt = 0


# # for i in range(100):
for i in range(check_episode):
    
    seed = np.random.randint(0, 500)
    
    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False
    total_reward = 0
    record = deque(maxlen=seq_length)
    cnt = 0
    prob = -1
    steps = 0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        if input_tag:
            record = transform_input(obs, action, model, record, alg_tag)
        else:
            record.append(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        cnt += 1
        if len(record) == 20:
            if input_tag:
                obs_input = torch.stack(list(record), dim=1)
            else:
                obs_input = torch.as_tensor(np.array(record), dtype=torch.float32).unsqueeze(0)
            obs_input = obs_input.permute(0, 2, 1).unsqueeze(0).float().to('cuda:0')
            a = todeynet(obs_input)
            # print(torch.softmax(a, dim=1)[0][1].item())
            label = torch.argmax(a, dim=1)
            if label.item() == 1 and steps==0:
                pre_label[i] = 1
                prob = torch.softmax(a, dim=1)[0][1].item()
                pic = Image.fromarray(env.render(mode='rgb_array'))
                image_filename = f'image_{save_cnt}.png'
                save_path = os.path.join(output_dir, image_filename)
                pic.save(save_path)
                state_save.append(obs)
                save_cnt += 1
                steps = cnt
    if steps == 0:
        prob = torch.softmax(a, dim=1)[0][1].item()
                
    if args.dataset == 'BipedalWalkerHC':
        if total_reward < 285:
            true_label[i] = 1 
    elif args.dataset == 'CartPoleS':
        if cnt < 200:
            true_label[i] = 1
    elif args.dataset == 'MountainCarS':
        if total_reward == -200:
            true_label[i] = 1
    else:   
        if cnt < 1000:
            true_label[i] = 1

    print("Episode: ", i, "Reward: ", total_reward, "Pre: ", pre_label[i], "True: ", true_label[i], "Prob: ", prob, 'Steps: ', cnt - steps)
    
    df.loc[len(df)] = [i, total_reward, pre_label[i], true_label[i], prob, cnt - steps]

state_save.save(output_dir + 'state_save.npy', state_save)

df.to_csv(result_save_dir, index=False)

    

