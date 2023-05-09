import csv

import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import seaborn as sn
import time
import torch.nn as nn
# import numba as nb
import os
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader


class parameters:
    n = 1000  # number of bidders
    t = 5000  # number of epchos
    m = 50  # number of queries
    embed = 8  # hidden layer
    beta = round(0.7 * n)
    lamb = 0.1
    lamb_rate = 0.005
    model_rate = 0.005   # trainer learning rate
    proportion_of_0 = 0.2
    space = 100
    d = 2  
    train = 1  
    integral_space = 100
    relax_budget = 0.2
    weight_q = 0.2


class Monotonic(nn.Module):
    def __init__(self):
        super().__init__()

        self.hyper_w_1 = nn.Sequential(nn.Linear(parameters.n, parameters.embed),
                                       nn.ReLU(),
                                       nn.Linear(parameters.embed, 1))
        self.hyper_w_final = nn.Sequential(nn.Linear(parameters.n, parameters.embed),
                                           nn.ReLU(),
                                           nn.Linear(parameters.embed, 1))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(parameters.n, 1)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(parameters.n, parameters.embed),
                               nn.ReLU(),
                               nn.Linear(parameters.embed, 1))

    def forward(self, theta):
        # First layer
        w1 = -torch.abs(self.hyper_w_1(theta))
        b1 = self.hyper_b_1(theta)
        hidden = w1 * theta + b1

        # Second layer
        w_final = torch.abs(self.hyper_w_final(theta))

        # State-dependent bias
        v = self.V(theta)

        # Compute final output
        y = hidden * w_final + v

        # Reshape and return
        return y.sigmoid() * 0.95


def trainset_to_x(m, n, trainset):
    flat_trainset = trainset.flatten()
    ones = np.ones(len(flat_trainset)).astype(np.float32)
    ts = np.linspace(flat_trainset, ones, parameters.integral_space).astype(np.float32)
    x = ts.T.reshape((m, n, parameters.integral_space))
    return x


def check_loss(loss_list):
    plot_df = pd.DataFrame([loss_list]).T
    plot_df.rename(columns={0: 'loss'}, inplace=True)
    sn.lineplot(data=plot_df)
    save_path = "C:/Users/ROG/Desktop/Monotonic/" + "n=" + str(parameters.n) + "m=" + str(parameters.m) + "train=" \
                + str(parameters.train) + "t=" + str(parameters.t) + "b=" + str(parameters.beta) + 'lamb=' \
                + str(parameters.lamb) + "lamb_rate=" + str(parameters.lamb_rate) + "/"
    plt.savefig(save_path + "Loss")


def check_obj(obj_list):
    plot_df = pd.DataFrame([obj_list]).T
    plot_df.rename(columns={0: 'objective'}, inplace=True)
    sn.lineplot(data=plot_df)
    save_path = "C:/Users/ROG/Desktop/Monotonic/" + "n=" + str(parameters.n) + "m=" + str(parameters.m) + "train=" \
                + str(parameters.train) + "t=" + str(parameters.t) + "b=" + str(parameters.beta) + 'lamb=' \
                + str(parameters.lamb) + "lamb_rate=" + str(parameters.lamb_rate) + "/"
    plt.savefig(save_path + "Obj")


def train():
    start = time.time()
    save_path = "C:/Users/ROG/Desktop/Monotonic/" + "n=" + str(parameters.n) + "m=" + str(parameters.m) + "train=" \
                 + str(parameters.train) + "t=" + str(parameters.t) + "b=" + str(parameters.beta) + 'lamb=' \
                 + str(parameters.lamb) + "lamb_rate=" + str(parameters.lamb_rate) + "model_rate=" \
                + str(parameters.model_rate) + "embed=" + str(parameters.embed) + "relax=" + str(parameters.relax_budget) \
                    + "weight=" + str(parameters.weight_q) + "/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        print("File exists.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # generate train set (theta)
    trainset = torch.sort(torch.rand(parameters.m, parameters.n, device=device))[0]
    trainset_np = trainset.to('cpu')
    x = torch.tensor(trainset_to_x(parameters.m, parameters.n, trainset=trainset_np.numpy())).to(device)
    x = torch.transpose(x, 1, 2)

    train_data = trainset_np.numpy()
    pd.DataFrame(train_data).to_csv(save_path + str(int(parameters.beta)) + "," + str(parameters.proportion_of_0) + ' trainset.csv')

    # train the network model
    lamb = torch.full((parameters.m,), parameters.lamb).to(device)
    net = Monotonic()
    net.to(device)
    trainer = torch.optim.SGD(net.parameters(), lr=parameters.model_rate)
    scheduler = lr_scheduler.StepLR(trainer, step_size=500, gamma=0.5)
    obj_list = []
    m_obj_list = [] 
    output_list = []
    loss_list = []
    cons_list = []
    m_cons_list = []
    lag_list = []
    lamb_list = []

    for iter in range(parameters.t):
        # train model
        iteration = 0
        while iteration < parameters.train:

            output = net(trainset)  # q
            obj = -((output * torch.log((1+output)/(1-output))).sum(1))  # the total privacy: q*log((1+q)/(1-q))

            y0 = net(x)
            y = y0 * torch.log((1 + y0) / (1 - y0))
            z = torch.trapezoid(torch.transpose(y, 2, 1), torch.transpose(x, 2, 1))
            integral = z.sum(1)     # int q*log((1+q)/(1-q))
                        
            cons = (trainset * output * torch.log((1 + output) / (1 - output))).sum(1) + integral - (1 + parameters.relax_budget) * parameters.beta
            lag = torch.sum(obj + lamb * cons) + parameters.weight_q * torch.sum(output)
            loss = torch.exp(lag / (parameters.m * parameters.n))

            obj_list.append(float(torch.mean(-obj)))
            m_obj_list.append(float(torch.max(-obj)))   # the highest obj among all exp
            output_list.append(float(torch.max(output)))     # the highest q among all agents and exp

            cons_list.append(float(torch.mean(cons)))   # should be negative
            m_cons_list.append(float(torch.max(cons)))
            lag_list.append(float(lag))
            loss_list.append(float(loss))
            
            if iter == 0:
                obj_best = torch.mean(-obj)
                inte_best = integral

                obj_best_bf = torch.mean(-obj)
                inte_best_bf = integral

                obj_bf = torch.mean(-obj)
                inte_bf = integral

                torch.save(net, save_path + "/best_model")
                torch.save(net, save_path + "/best_bf_model")
                torch.save(net, save_path + "/bf_model")
            else:
                if torch.mean(-obj) >= obj_best:
                    obj_best = torch.mean(-obj)
                    inte_best = integral
                    torch.save(net, save_path + "/best_model")
                if (torch.mean(-obj) >= obj_best_bf) and (torch.mean(cons) <= 0):
                    obj_best_bf = torch.mean(-obj)
                    inte_best_bf = integral
                    torch.save(net, save_path + "/best_bf_model")
                if (torch.mean(-obj) >= obj_bf) and (torch.max(cons) <= 0):
                    obj_bf = torch.mean(-obj)
                    inte_bf = integral
                    torch.save(net, save_path + "/bf_model")
            
            trainer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)  #clip parameter
            trainer.step()
            scheduler.step()

            torch.cuda.empty_cache()
            iteration += 1

        if (iter + 1) % 1000 == 0:
            print("Loss for iteration %d is %f" % (iter + 1, loss))
            pass
        torch.cuda.empty_cache()

        # update lambda
        q_star = net.forward(trainset).detach()
        y0 = net(x)
        y = y0 * torch.log((1 + y0) / (1 - y0))
        z = torch.trapezoid(torch.transpose(y, 2, 1), torch.transpose(x, 2, 1))
        integral_star = z.sum(1)
        

        lamb = lamb + parameters.lamb_rate * ((trainset * q_star * torch.log((1+q_star)/(1-q_star))).sum(1) +
                                              integral_star.detach() - (1 + parameters.relax_budget) * parameters.beta)
        for i in range(len(lamb)):
            if lamb[i] <= 0:
                lamb[i] = 1e-4
        lamb_list.append(float(torch.mean(lamb)))

        torch.cuda.empty_cache()

    # save and return the trained model
    end = time.time()
    print("time: ", end-start)

   
    last_allocation = net(trainset).detach()
    last_obj = ((last_allocation * torch.log((1+last_allocation)/(1-last_allocation))).sum(1))
    last_cons = (trainset * last_allocation * torch.log((1+last_allocation)/(1-last_allocation))).sum(1) + integral
   
    print("(Trainset) (last) Total (expected) privacy: ", torch.mean(last_obj))
    print("(Trainset) (last) Total (expected) payment: ", torch.mean(last_cons))
    
    best_model = torch.load(save_path + 'best_model')
    best_allocation = best_model(trainset).detach()
    best_obj = ((best_allocation * torch.log((1+best_allocation)/(1-best_allocation))).sum(1))
    best_cons = (trainset * best_allocation * torch.log((1+best_allocation)/(1-best_allocation))).sum(1) + inte_best
    
    print("(Trainset) (best) Total (expected) privacy: ", torch.mean(best_obj))
    print("(Trainset) (best) Total (expected) payment: ", torch.mean(best_cons))
    
    best_bf_model = torch.load(save_path + 'best_bf_model')
    best_allocation_bf = best_bf_model(trainset).detach()
    best_obj_bf = ((best_allocation_bf * torch.log((1+best_allocation_bf)/(1-best_allocation_bf))).sum(1))
    best_cons_bf = (trainset * best_allocation_bf * torch.log((1+best_allocation_bf)/(1-best_allocation_bf))).sum(1) \
                    + inte_best_bf
    
    print("(Trainset) (best bf) Total (expected) privacy: ", torch.mean(best_obj_bf))
    print("(Trainset) (best bf) Total (expected) payment: ", torch.mean(best_cons_bf))
    print("(Trainset) (best bf) Total (expected) payment: ", best_cons_bf)

    bf_model = torch.load(save_path + 'bf_model')
    allocation_bf = bf_model(trainset).detach()
    obj_bf = ((allocation_bf * torch.log((1+allocation_bf)/(1-allocation_bf))).sum(1))
    cons_bf = (trainset * allocation_bf * torch.log((1+allocation_bf)/(1-allocation_bf))).sum(1) \
                    + inte_bf
    
    print("(Trainset) (bf) Total (expected) privacy: ", torch.mean(obj_bf))
    print("(Trainset) (bf) Total (expected) payment: ", cons_bf)
    print(torch.cuda.max_memory_allocated())

    torch.save(net, save_path + "/model")
    torch.save(loss_list, save_path + "/loss")
    torch.save(obj_list, save_path + "/mean_obj")
    torch.save(m_obj_list, save_path + "/max_obj")
    torch.save(output_list, save_path + "/allocation")
    torch.save(cons_list, save_path + "/constraint")
    torch.save(m_cons_list, save_path + "/max constraint")
    torch.save(lag_list, save_path + "/lag")
    torch.save(lamb_list, save_path + "/lambda")
    torch.save(trainset, save_path + "/trainset")

    with open(save_path + "/obj,mobj,cons,lag,loss,lambda.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Interation', 'mean obj', 'max obj', 'constraint', 'max cons', 'lag', "loss", "lambda"])
        for i in range(parameters.t):
            writer.writerow([i, obj_list[i], m_obj_list[i], cons_list[i], m_cons_list[i], lag_list[i], loss_list[i], lamb_list[i]])

    return net


train()
