import os

import torch
import torch.optim as optim
from utils import *
import model
from evaluation import *
from collections import defaultdict



"""
The meaning of parameters:
self.dataset: Which dataset is used to train the model? Such as 'FB15k', 'WN18', etc.
self.learning_rate: Initial learning rate (lr) of the model.
self.embedding_size: The embedding size of entities and relations.
self.num_batches: How many batches to train in one epoch?
self.train_iters: The maximum number of epochs for training.
self.filter: Whether to check a generated negative sample is false negative.
self.optimizer: Which optimizer to use? Such as SGD, Adam, etc.
self.entity_total: The number of different entities.
self.relation_total: The number of different relations.
"""


class Config(object):
    def __init__(self):
        self.filter = True
        self.optimizer = optim.Adam
        self.entity_total = 0
        self.relation_total = 0
        self.batch_size = 0

import argparse
argparser = argparse.ArgumentParser()

"""
The meaning of some parameters:
seed: Fix the random seed. Except for 0, which means no setting of random seed.
which is a demo showing training curves in real time.
You can refer to https://github.com/WarBean/hyperboard to know more.
num_processes: Number of processes used to evaluate the result.
"""

argparser.add_argument('-d', '--dataset', type=str)
argparser.add_argument('-l', '--learning_rate', type=float, default=0.01)
argparser.add_argument('-en', '--entity_function', type=int, default=2)
argparser.add_argument('-nb', '--num_batches', type=int, default=100)
argparser.add_argument('-n', '--train_iters', type=int, default=100)
argparser.add_argument('-f', '--filter', type=int, default=1)
argparser.add_argument('-s', '--seed', type=int, default=0)
argparser.add_argument('-op', '--optimizer', type=int, default=1)
argparser.add_argument('-np', '--num_processes', type=int, default=4)
# for regularizing
argparser.add_argument('-lam', '--lmbda', type=float, default=0.001)
# for tradeoff
argparser.add_argument('-lam2', '--trade_lambda', type=float, default=1)
argparser.add_argument('-nf', '--feature_size', type=int, default=100)
argparser.add_argument('-hid1', '--hidden1', type=int, default=100)
argparser.add_argument('-hid2', '--hidden2', type=int, default=100)
argparser.add_argument('-drop', '--dropout', type=float, default=0.5)
argparser.add_argument('-basis_w', '--num_basis_w', type=int, default=20)
argparser.add_argument('-basis_e', '--num_basis_e', type=int, default=5)
argparser.add_argument('-evnt', '--evnt', type=int, default=1)
argparser.add_argument('-model', '--model', type=int, default=1)
args = argparser.parse_args()

if args.seed != 0:
    torch.manual_seed(args.seed)

trainTotal, trainList, trainDict, trainTimes = load_quadruples('./data/', 'train_500.txt')
testTotal, testList, testDict, testTimes = load_quadruples('./data/', 'test_500.txt')
quadrupleTotal, quadrupleList, quadrupleDict, _ = load_quadruples('./data/', 'train_500.txt', 'test_500.txt')
config = Config()
config.learning_rate = args.learning_rate
config.lmbda = args.lmbda
config.lmbda2 = args.trade_lambda
config.hidden1 = args.hidden1
config.hidden2 = args.hidden2
config.dropout = args.dropout
config.basis_w = args.num_basis_w
config.basis_e = args.num_basis_e
config.time_size = args.time_size
config.entity_total, config.relation_total = get_total_number('./data/', 'stat_500.txt')

if args.entity_function == 1:
    config.entity_function = True
else:
    config.entity_function = False

config.feature_size = args.feature_size
config.train_iters = args.train_iters

if args.filter == 1:
    config.filter = True
else:
    config.filter = False

if args.optimizer == 0:
    config.optimizer = optim.SGD
elif args.optimizer == 1:
    config.optimizer = optim.Adam
elif args.optimizer == 2:
    config.optimizer = optim.RMSprop

if args.evnt == 1:
    config.evnt = True
else:
    config.evnt = False


filename = '_'.join(
    ['md', str(args.model),
     'lam', str(args.lmbda),
     'l', str(args.learning_rate),
     'nf', str(args.feature_size),
     'nb', str(args.num_batches),
     'n', str(args.train_iters),
     'f', str(args.filter),
     's', str(args.seed),
     'op', str(args.optimizer),]) + '.ckpt'

path_name = os.path.join('./model/', filename)
if os.path.exists(path_name):
    model = torch.load(path_name)
elif args.model == 0:
    model = model.Yourmodel(config)



optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)

start_time = time.time()

model.train() # for dropout
for epoch in range(config.train_iters):
# for epoch in range(2):
    adj_lists_in = [defaultdict(set) for _ in range(config.relation_total)]
    adj_lists_out = [defaultdict(set) for _ in range(config.relation_total)]

    total_loss = 0
    optimizer.zero_grad()

    for tim in trainTimes:
        # update adj list
        # optimizer.zero_grad()
        quadruple_t = get_quadruple_t(quadrupleList, tim)
        adj_lists_in, adj_lists_out = update_adj_lists(adj_lists_in, adj_lists_out, quadruple_t)
        # print(adj_lists_out)

        loss = model.loss(quadruple_t, quadrupleDict, tim, adj_lists_in, adj_lists_out, train=True)
        total_loss += loss
        print("time: ", tim, "total loss: ", total_loss.item(), "loss: ", loss.item())
        # loss.backward(retain_graph=False)
        # optimizer.step()


    for param in model.parameters():
        if param.grad is not None:
            print(torch.sum(torch.sum(param.grad.data)))

    # start debugger
    # import pdb; pdb.set_trace()
    print("total loss: ", total_loss.item())
    total_loss.backward()
    # total_loss.backward(retain_graph=True)
    optimizer.step()

    final_adj_in = adj_lists_in
    final_adj_out = adj_lists_out

# print(final_adj_in)
hit_t = []
mrr_t = []
model.eval()
for time in testTimes:
    test = get_quadruple_t(testList, time)
    # add time decay factor
    ent_output, rel_output = model(testList, time, final_adj_in, final_adj_out, train=False)
    # print(torch.sum(torch.sum(ent_output)))
    hit10, mrr = evaluation(test, quadrupleDict, ent_output.detach().numpy(), rel_output.detach().numpy(), filter=True, head=0)
    hit_t.append(hit10)
    mrr_t.append(mrr)

with open('result.txt', 'w') as f:
    f.write("hit\n")
    f.write("%s\n" % hit_t)
    f.write("%s\n" %(sum(hit_t)/float(len(hit_t))))
    f.write("mr\n")
    f.write("%s\n" % mrr_t)
    f.write("%s\n" % (sum(mrr_t) / float(len(mrr_t))))