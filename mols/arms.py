import random
import torch
import numpy as np
import os
import csv
import heapq

class Arm:
    def __init__(self, mu: float, sigma: float):
        """ Mean and standard deviation for the normal distribution."""
        self.mu = mu
        self.sigma = sigma
        self.T = 0

    def draw(self, t):
        """ Returns the achieved reward of the arm at this round. """
        # return random.gauss(self.mu, self.sigma) + (3*torch.tensor(t, dtype=torch.float)/(2*self.T)).pow(0.5)
        return self.mu + (3*torch.log(torch.tensor(t, dtype=torch.float))/(2*self.T)).pow(0.5)

    def update(self, reward: float):
        """ Updates the mean and standard deviation of the arm. """
        self.T += 1
        self.mu = (self.mu * (self.T - 1) + reward) / self.T
        self.sigma = ((self.sigma ** 2 * (self.T - 1) + (reward - self.mu) ** 2) / self.T) ** 0.5

class Oracle:
    def __init__(self, args, K,num_elements,interval=400):
        self.num_elements=num_elements
        self.interval = interval
        self.K = K
        self.explore_cnt = 0
        self.t=1
        self.arms = [Arm(0.1, 0.01) for _ in range(self.num_elements)]
        self.history = [0]
        self.block_sum_scores = [0 for _ in range(self.num_elements)]
        self.block_cnts = [0 for _ in range(self.num_elements)]
        self.init_ok = False
        self.init_cnt = 20
        self.acc_regret = []
        self.total_scores = []
        self.args = args
        exp_dir = f'{self.args.save_path}/{self.args.partial}_{self.args.objective}_{K}_{self.args.random}_new/'
        os.makedirs(exp_dir, exist_ok=True)
        self.csv_file = f'{exp_dir}/rewards_results_final.csv'
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['r1', 'r2'])
        self.oracle()
    
    def oracle(self):
        print('stage is:', self.init_ok, self.t, len(self.history))
        if not self.init_ok:
            init_ok = True
            for arm in self.arms:
                if arm.T == 0:
                    init_ok = False
            if self.t < self.init_cnt:
                init_ok = False
            self.init_ok = init_ok
        if not self.init_ok:
            self.masks = torch.tensor(np.random.choice([True, False], size=self.num_elements, p=[1, 0]) )
        else:
            if self.args.random ==1:
                sample_indices = torch.randperm(self.num_elements)[:self.K]
                print('random idxs are: ',sample_indices)
            else:
                mu_s = [arm.draw(self.t) for arm in self.arms]
                scores_tensor = torch.tensor(mu_s, dtype=torch.float32)
                _, sample_indices = torch.topk(scores_tensor, self.K)
            self.sample_indices = sample_indices
            self.masks=torch.zeros(self.num_elements, dtype=torch.bool)
            self.masks[sample_indices] = True
        self.true_indices = torch.nonzero(self.masks).squeeze() 
        self.explore_cnt = 0
        self.sampled_mols = []


    def update(self,traj):
        real_r, m, trajectory_stats, inflow=traj
        for idxs in m.blockidxs:
            self.block_sum_scores[idxs]+=real_r
            self.block_cnts[idxs]+=1
        self.sampled_mols.append(traj)
        self.explore_cnt+=1
    
    def calc_avg(self):
        sz = min(20,len(self.history))
        return (self.history[-1]-self.history[-sz])/sz

    def update_arms(self):
        tot_rewards, tot_cnts = 0, 0
        for idxs in range(self.num_elements):
            if self.block_cnts[idxs] > 0:
                self.arms[idxs].update((self.block_sum_scores[idxs]) / self.block_cnts[idxs])
                tot_rewards+=(self.block_sum_scores[idxs]) / self.block_cnts[idxs]
                tot_cnts+=1
        tot_rewards/=tot_cnts
        self.block_cnts = [0 for _ in range(self.num_elements)]
        self.block_sum_scores = [0 for _ in range(self.num_elements)]
        self.explore_cnt = 0
        self.t += 1
        self.total_scores.append(tot_rewards)
        self.acc_regret.append(max(self.total_scores[-1000:])-self.total_scores[-1] + (self.acc_regret[-1] if len(self.acc_regret) else 0))
        mus = [arm.mu for arm in self.arms]
        optu = sum(heapq.nlargest(self.K, mus))
        if hasattr(self, 'sample_indices'):
            curru = sum(mus[i] for i in self.sample_indices.tolist())
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([optu,curru])
        print('accumulate regret is: ',self.acc_regret[-1])
        
    
    def update_history(self,new_his):
        self.history.append(new_his)



