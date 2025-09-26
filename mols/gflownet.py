import argparse
from copy import copy, deepcopy
from collections import defaultdict
from datetime import timedelta
import gc
import gzip
import os
import os.path as osp
import pickle
import psutil
import pdb
import subprocess
import sys
import threading
import time
import traceback
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn

from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
import model_atom, model_block, model_fingerprint
import random
from metrics import Evaluator
from arms import Oracle
import csv
import copy
import psutil, GPUtil

tmp_dir = "/tmp/molexp"
os.makedirs(tmp_dir, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=5e-4, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=4, help="Minibatch size", type=int)
parser.add_argument("--opt_beta", default=0.9, type=float)
parser.add_argument("--opt_beta2", default=0.999, type=float)
parser.add_argument("--opt_epsilon", default=1e-8, type=float)
parser.add_argument("--nemb", default=256, help="#hidden", type=int)
parser.add_argument("--min_blocks", default=2, type=int)
parser.add_argument("--max_blocks", default=8, type=int)
parser.add_argument("--num_iterations", default=250000, type=int)
parser.add_argument("--num_conv_steps", default=10, type=int)
parser.add_argument("--log_reg_c", default=2.5e-5, type=float)
parser.add_argument("--reward_exp", default=10, type=float)
parser.add_argument("--reward_norm", default=8, type=float)
parser.add_argument("--sample_prob", default=1, type=float)
parser.add_argument("--R_min", default=0.1, type=float)
parser.add_argument("--leaf_coef", default=10, type=float)
parser.add_argument("--clip_grad", default=0, type=float)
parser.add_argument("--clip_loss", default=0, type=float)
parser.add_argument("--replay_mode", default='online', type=str)
parser.add_argument("--bootstrap_tau", default=0, type=float)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--random_action_prob", default=0.05, type=float)
parser.add_argument("--array", default='')
parser.add_argument("--repr_type", default='block_graph')
parser.add_argument("--model_version", default='v4')
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--save_path", default='results/')
parser.add_argument("--proxy_path", default='./data/pretrained_proxy')
parser.add_argument("--print_array_length", default=False, action='store_true')
parser.add_argument("--progress", default='yes')
parser.add_argument("--floatX", default='float64')
parser.add_argument("--include_nblocks", default=False)
parser.add_argument("--balanced_loss", default=True)
# If True this basically implements Buesing et al's TreeSample Q,
# samples uniformly from it though, no MTCS involved
parser.add_argument("--do_wrong_thing", default=False)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--partial", default=0, type=int)
parser.add_argument("--do_uniform", default=0, type=int)
parser.add_argument("--objective", default='fm', type=str)
parser.add_argument("--K", default= 50 , type=int)
parser.add_argument("--random", default= 0 , type=int)

state_node_map = {} # smiles -> node
c_puct = 1 
n_playout = 1  
max_deepth = 10
epsilon = 0.2 
# initial_epsilon = 0 
# final_epsilon = 0.1
# transition_step = 1000


def set_seed(
    seed: int, deterministic_torch: bool = False
):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    # torch_geometric.seed_everything(seed)
    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(True)

class TreeNode:
    def __init__(self, mol, is_terminal):
        get = lambda x, d: getattr(args, x) if hasattr(args, x) else d
        self._parent = {} 
        self._children = {} 
        self._n_visits = 0
        self.max_blocks = get('max_blocks', 10)
        self.min_blocks = get('min_blocks', 2)
        self._Q = 0  
        self.mol = mol
        self.puct_score = {}  
        self.is_terminal = is_terminal 

    def is_fully_expanded(self):
        if self._children:
            return True
        return False

    def select(self):
        if self.is_terminal:
            return self, -2
        if not self.is_fully_expanded():
            return self, -1  

        actions = list(self._children.keys())
        puct_values = np.array([self.get_value(c_puct, a) for a in actions])
        exp_values = np.exp(puct_values - np.max(puct_values))  
        probs = exp_values / exp_values.sum()
        selected_idx = np.random.choice(len(actions), p=probs)
        return self._children[actions[selected_idx]], actions[selected_idx]

    def get_value(self, c_puct, a): 
        num_actions = len(self._children) 
        p = 1.0 / num_actions        
        u = c_puct * p * np.sqrt(self._n_visits) / (1 + self._children[a]._n_visits)
        # return (self._children[a]._Q) + u**5
        return u

    def update(self, reward):
        self._n_visits += 1
        self._Q += (reward - self._Q) / self._n_visits
        
    def update_recursive(self, reward, path):
        for node in reversed(path):
            if node == '':
                break
            if node is not None:
                node.update(reward)
    
def smiles2inchi(smiles):
    mol = Chem.MolFromSmiles(smiles)
    inchi = Chem.MolToInchiKey(mol)
    return inchi

class Dataset:
    def __init__(self, args, bpath, device, floatX=torch.double):
        self.test_split_rng = np.random.RandomState(142857)
        self.train_rng = np.random.RandomState(142857)
        self.train_mols = []
        self.test_mols = []
        self.train_mols_map = {}
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, args.repr_type, include_nblocks=args.include_nblocks)
        self.mdp.build_translation_table()
        self._device = device
        self.seen_molecules = set()
        self.stop_event = threading.Event()
        self.target_norm = [-8.6, 1.10]
        self.sampling_model = None
        self.sampling_model_prob = 0
        self.floatX = floatX
        self.mdp.floatX = self.floatX
        #######
        # This is the "result", here a list of (reward, BlockMolDataExt, info...) tuples
        self.sampled_mols = []

        get = lambda x, d: getattr(args, x) if hasattr(args, x) else d
        self.min_blocks = get('min_blocks', 2)
        self.max_blocks = get('max_blocks', 10)
        self.mdp._cue_max_blocks = self.max_blocks
        self.replay_mode = get('replay_mode', 'dataset')
        self.reward_exp = get('reward_exp', 1)
        self.reward_norm = get('reward_norm', 1)
        self.random_action_prob = get('random_action_prob', 0)
        self.R_min = get('R_min', 1e-8)
        self.do_wrong_thing = get('do_wrong_thing', False)

        self.online_mols = []
        self.max_online_mols = 1000
        self.evaluator = Evaluator()
        self.last_idx = 0
        self.num_elements=105
        self.oracle = Oracle(args,args.K,self.num_elements)
        self.args = args


    
    def _get(self, i, dset,current_step):
        if ((self.sampling_model_prob > 0 and # don't sample if we don't have to
             self.train_rng.uniform() < self.sampling_model_prob)
            or len(dset) < 32):
                return self._get_sample_model(current_step)
        # Sample trajectories by walking backwards from the molecules in our dataset

        # Handle possible multithreading issues when independent threads
        # add/substract from dset:
        while True:
            try:
                m = dset[i]
            except IndexError:
                i = self.train_rng.randint(0, len(dset))
                continue
            break
        if not isinstance(m, BlockMoleculeDataExtended):
            m = m[-1]
        r = m.reward
        done = 1
        samples = []
        # a sample is a tuple (parents(s), parent actions, reward(s), s, done)
        # an action is (blockidx, stemidx) or (-1, x) for 'stop'
        # so we start with the stop action, unless the molecule is already
        # a "terminal" node (if it has no stems, no actions).
        if len(m.stems):
            samples.append(((m,), ((-1, 0),), r, m, done))
            r = done = 0
        while len(m.blocks): # and go backwards
            parents, actions = zip(*self.mdp.parents(m))
            samples.append((parents, actions, r, m, done))
            r = done = 0
            m = parents[self.train_rng.randint(len(parents))]
        return samples

    def set_sampling_model(self, model, proxy_reward, sample_prob=0.5):
        self.sampling_model = model
        self.sampling_model_prob = sample_prob
        self.proxy_reward = proxy_reward

    def _get_sample_model_balance(self):
        m = BlockMoleculeDataExtended()
        samples = []
        max_blocks = self.max_blocks
        trajectory_stats = []
        for t in range(max_blocks):
            s = self.mdp.mols2batch([self.mdp.mol2repr(m)])
            s_o, m_o = self.sampling_model(s)
            ## fix from run 330 onwards
            if t < self.min_blocks:
                m_o = m_o * 0 - 1000 # prevent assigning prob to stop
                                     # when we can't stop
            ##
            logits = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
            cat = torch.distributions.Categorical(
                logits=logits)
            action = cat.sample().item()
            if self.random_action_prob > 0 and self.train_rng.uniform() < self.random_action_prob:
                action = self.train_rng.randint(int(t < self.min_blocks), logits.shape[0])

            q = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
            trajectory_stats.append((q[action].item(), action, torch.logsumexp(q, 0).item()))
            if t >= self.min_blocks and action == 0:
                r = self._get_reward(m)
                samples.append(((m,), ((-1,0),), r, m, 1))
                break
            else:
                # action = max(0, action-1)
                # action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                action = max(0, action-1)
                action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                m_old = m
                m = self.mdp.add_block_to(m, *action)
                if len(m.blocks) and not len(m.stems) or t == max_blocks - 1:
                    # can't add anything more to this mol so let's make it
                    # terminal. Note that this node's parent isn't just m,
                    # because this is a sink for all parent transitions
                    r = self._get_reward(m)
                    if self.do_wrong_thing:
                        samples.append(((m_old,), (action,), r, m, 1))
                    else:
                        samples.append((*zip(*self.mdp.parents(m)), r, m, 1))
                    break
                else:
                    if self.do_wrong_thing:
                        samples.append(((m_old,), (action,), 0, m, 0))
                    else:
                        samples.append((*zip(*self.mdp.parents(m)), 0, m, 0))
        p = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in samples[-1][0]])
        qp = self.sampling_model(p, None)
        qsa_p = self.sampling_model.index_output_by_action(
            p, qp[0], qp[1][:, 0],
            torch.tensor(samples[-1][1], device=self._device).long())
        inflow = torch.logsumexp(qsa_p.flatten(), 0).item()
        self.oracle.update(traj=(r, m, trajectory_stats, inflow))
    
    def evaluate_rollout(self, m):
        cumulative_reward = 0
        for i in range(max_deepth):  
            s = self.mdp.mols2batch([self.mdp.mol2repr(m)])
            s_o, m_o = self.sampling_model(s)
            if m.numblocks < self.min_blocks:
                    m_o = m_o * 0 - 1000
            logits = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
            cat = torch.distributions.Categorical(logits=logits)
            action = cat.sample().item()
            if self.random_action_prob > 0 and self.train_rng.uniform() < self.random_action_prob:
                action = self.train_rng.randint(int(m.numblocks < self.min_blocks), logits.shape[0])
            if m.numblocks >= self.min_blocks and action == 0:
                r = self._get_reward(m)
                # print(m.smiles, "reward:", r)
                cumulative_reward += r
                break
            else:
                action = max(0, action-1)
                action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                # print(m.stems, " ",m.smiles,'    ',*action)
                m = self.mdp.add_block_to(m, *action)
                if len(m.blocks) and not len(m.stems) or m.numblocks == self.max_blocks:
                    r = self._get_reward(m)
                    # print(m.smiles, "reward:", r)
                    cumulative_reward += r
                    break
                # print(m.smiles, "reward", 0)
    
        return cumulative_reward

    def playout(self, m, path):
        # m_copy = copy.deepcopy(m)
        node = state_node_map[m.smiles]
        count = 0
    # SELECT
        while True:
            best_node, best_action = node.select()
            if best_action == -2:
                reward = self._get_reward(best_node.mol)
                best_node.update_recursive(reward, path)
                for _ in range(count):
                    path.pop()
                return 
            if best_action == -1:
                break
      
            path.append(best_node)
            count += 1
            # best_action = max(0, best_action-1)
            # best_action = (best_action % self.mdp.num_blocks, best_action // self.mdp.num_blocks)
            # if(node.mol.smiles != m_copy.smiles):
            #     print("node.smiles != m_copy.smiles")
            #     print("node.smiles:",node.mol.smiles,"m_copy.smiles:",m_copy.smiles)
            #     exit(0)
            # m_copy = self.mdp.add_block_to(m_copy, *best_action)
            node = best_node
    # EXPANSION
        m_copy = copy.deepcopy(node.mol)
        s = self.mdp.mols2batch([self.mdp.mol2repr(m_copy)])
        s_o, m_o = self.sampling_model(s)
        flag = 0
        if m_copy.numblocks < self.min_blocks:
                flag = 1
                m_o = m_o * 0 - 1000
        logits = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
        action_probs = torch.softmax(logits, dim=0)
        for i, prob in enumerate(action_probs):
            m_temp = copy.deepcopy(m_copy)
            if i!=0:
                i_temp=max(0, i-1)
                i_temp = (i_temp % self.mdp.num_blocks, i_temp // self.mdp.num_blocks)
                m_temp = self.mdp.add_block_to(m_temp, *i_temp)
                if m_temp.smiles in state_node_map:
                    child = state_node_map[m_temp.smiles]
                else:
                    if (len(m_temp.blocks) and not len(m_temp.stems)) or m_temp.numblocks == self.max_blocks:
                        child = TreeNode(m_temp, True)
                    else:
                        child = TreeNode(m_temp, False)
                    state_node_map[m_temp.smiles] = child
                node._children[i] = child
            else:
                if flag == 0:
                    child = TreeNode(m_temp, True)
                    node._children[i] = child
    # SIMULATION
        # selected_id = torch.multinomial(action_probs, num_samples=1).item()
        cat = torch.distributions.Categorical(logits=logits)
        selected_id = cat.sample().item()
        if node.mol.numblocks >= self.min_blocks and selected_id == 0:
            leaf_value = self._get_reward(node.mol)
            path.append(node._children[0])
            node.update_recursive(leaf_value, path)
            for _ in range(count+1):
                if path:
                    path.pop()
            return 
        else:
            old_selected_id = selected_id
            selected_id = max(0,selected_id-1)
            action = (selected_id % self.mdp.num_blocks, selected_id // self.mdp.num_blocks)
            m_copy = self.mdp.add_block_to(m_copy, *action)
            path.append(state_node_map[m_copy.smiles])
            if len(m_copy.blocks) and not len(m_copy.stems) or m_copy.numblocks == self.max_blocks:
                leaf_value = self._get_reward(m_copy)
                (state_node_map[m_copy.smiles]).update_recursive(leaf_value, path)
                for _ in range(count+1):
                    if path:
                        path.pop()
                return 
            leaf_value = self.evaluate_rollout(m_copy)
            (node._children[old_selected_id]).update_recursive(leaf_value, path)
            for _ in range(count+1):
                if path:
                    path.pop()

    def get_mcts_weight(self, current_step):
        if current_step >= transition_step:
            return final_epsilon
        
        decay = final_epsilon - initial_epsilon
        return initial_epsilon + decay * (current_step / transition_step)

    def get_move(self, n_playout, path, m, current_step):
        current_node = state_node_map[m.smiles]
        s = self.mdp.mols2batch([self.mdp.mol2repr(current_node.mol)])
        s_o, m_o = self.sampling_model(s)
        if m.numblocks < self.min_blocks:
                m_o = m_o * 0 - 1000
        logits = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
        pf = torch.softmax(logits, dim=0)
        for _ in range(n_playout):
            self.playout(m, path)
        if not current_node._children:
            return 0
        q_values = torch.tensor(
            [(child._Q)**5 for _, child in current_node._children.items()],
            device=pf.device, dtype=pf.dtype
        )
        q_min = q_values.min()
        q_max = q_values.max()
        normalized_q = (q_values - q_min) / (q_max - q_min + 1e-8)  
        probs = normalized_q / normalized_q.sum()
        available_actions = list(current_node._children.keys())
        pf = pf[available_actions]
        # epsilon = self.get_mcts_weight(current_step)
        u = (1 - epsilon) * pf + epsilon * probs
        u = u / u.sum()
        selected_action_index = torch.multinomial(u, num_samples=1).item()
        return available_actions[selected_action_index]
        
    def _get_sample_model(self, current_step):
        m = BlockMoleculeDataExtended()
        samples = []
        max_blocks = self.max_blocks
        trajectory_stats = []
        path = []
        path.append(state_node_map[m.smiles])
        for t in range(max_blocks):
            s = self.mdp.mols2batch([self.mdp.mol2repr(m)])
            s_o, m_o = self.sampling_model(s)
            ## fix from run 330 onwards
            if t < self.min_blocks:
                m_o = m_o * 0 - 1000 # prevent assigning prob to stop
                                     # when we can't stop
            ##
            if self.args.partial:
                mask_s = s_o[:,self.oracle.masks]
                s_o= mask_s #105*n  50*n
            logits = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
            
            cat = torch.distributions.Categorical(
                logits=logits)
            # action = cat.sample().item()

            action = self.get_move(n_playout, path, m, current_step)
            action_id = action
            if self.random_action_prob > 0 and self.train_rng.uniform() < self.random_action_prob:
                action = self.train_rng.randint(int(t < self.min_blocks), logits.shape[0])
            q = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
            trajectory_stats.append((q[action].item(), action, torch.logsumexp(q, 0).item()))
            if t >= self.min_blocks and action == 0:
                r = self._get_reward(m)
                samples.append(((m,), ((-1,0),), r, m, 1))
                break
            else:
                action = max(0, action-1)
                if self.args.partial:
                    action = (self.oracle.true_indices[action % self.oracle.true_indices.shape[0] ].item(), action // self.oracle.true_indices.shape[0] )
                else:
                    action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                m_old = m
                # m = self.mdp.add_block_to(m, *action) 
                m = state_node_map[m.smiles]._children[action_id].mol 
                path.append(state_node_map[m.smiles])
                if len(m.blocks) and not len(m.stems) or t == max_blocks - 1:
                    # can't add anything more to this mol so let's make it
                    # terminal. Note that this node's parent isn't just m,
                    # because this is a sink for all parent transitions
                    r = self._get_reward(m)
                    if self.do_wrong_thing:
                        samples.append(((m_old,), (action,), r, m, 1))
                    else:
                        samples.append((*zip(*self.mdp.parents(m)), r, m, 1))
                    break
                else:
                    if self.do_wrong_thing:
                        samples.append(((m_old,), (action,), 0, m, 0))
                    else:
                        samples.append((*zip(*self.mdp.parents(m)), 0, m, 0))
        p = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in samples[-1][0]])
        qp = self.sampling_model(p, None)
        qsa_p = self.sampling_model.index_output_by_action(
            p, qp[0], qp[1][:, 0],
            torch.tensor(samples[-1][1], device=self._device).long())
        inflow = torch.logsumexp(qsa_p.flatten(), 0).item()
        self.sampled_mols.append((r, m, trajectory_stats, inflow))
        if self.args.do_uniform==0:
            self.oracle.update(traj=(r, m, trajectory_stats, inflow))
        else:
            self._get_sample_model_balance()
        if self.replay_mode == 'online' or self.replay_mode == 'prioritized':
            m.reward = r
            self._add_mol_to_online(r, m, inflow)
        return samples

    def _add_mol_to_online(self, r, m, inflow):
        if self.replay_mode == 'online':
            r = r + self.train_rng.normal() * 0.01
            if len(self.online_mols) < self.max_online_mols or r > self.online_mols[0][0]:
                self.online_mols.append((r, m))
            if len(self.online_mols) > self.max_online_mols:
                self.online_mols = sorted(self.online_mols)[max(int(0.05 * self.max_online_mols), 1):]
        elif self.replay_mode == 'prioritized':
            self.online_mols.append((abs(inflow - np.log(r)), m))
            if len(self.online_mols) > self.max_online_mols * 1.1:
                self.online_mols = self.online_mols[-self.max_online_mols:]


    def _get_reward(self, m):
        rdmol = m.mol
        if rdmol is None:
            return self.R_min
        smi = m.smiles
        if smi in self.train_mols_map:
            return self.train_mols_map[smi].reward
        return self.r2r(normscore=self.proxy_reward(m))

    def sample(self, n, current_step):
        if self.replay_mode == 'dataset':
            eidx = self.train_rng.randint(0, len(self.train_mols), n)
            samples = sum((self._get(i, self.train_mols) for i in eidx), [])
        elif self.replay_mode == 'online':
            eidx = self.train_rng.randint(0, max(1,len(self.online_mols)), n)
            samples = sum((self._get(i, self.online_mols, current_step) for i in eidx), [])
        elif self.replay_mode == 'prioritized':
            if not len(self.online_mols):
                samples = sum((self._get(0, self.online_mols) for i in range(n)), [])
            else:
                prio = np.float32([i[0] for i in self.online_mols])
                eidx = self.train_rng.choice(len(self.online_mols), n, False, prio/prio.sum())
                samples = sum((self._get(i, self.online_mols) for i in eidx), [])
        return zip(*samples)

    def sample2batch(self, mb):
        p, a, r, s, d, *o = mb
        mols = (p, s)
        # The batch index of each parent
        p_batch = torch.tensor(sum([[i]*len(p) for i,p in enumerate(p)], []),
                               device=self._device).long()
        # Convert all parents and states to repr. Note that this
        # concatenates all the parent lists, which is why we need
        # p_batch
        p = self.mdp.mols2batch(list(map(self.mdp.mol2repr, sum(p, ()))))
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])
    
        a = torch.tensor(sum(a, ()), device=self._device).long()
        # rewards and dones
        r = torch.tensor(r, device=self._device).to(self.floatX)
        d = torch.tensor(d, device=self._device).to(self.floatX)
        return (p, p_batch, a, r, s, d, mols, *o)

    def r2r(self, dockscore=None, normscore=None):
        if dockscore is not None:
            normscore = 4-(min(0, dockscore)-self.target_norm[0])/self.target_norm[1]
        normscore = max(self.R_min, normscore)
        return (normscore/self.reward_norm) ** self.reward_exp

    def start_samplers(self, n, mbsize):
        self.ready_events = [threading.Event() for i in range(n)]
        self.resume_events = [threading.Event() for i in range(n)]
        self.results = [None] * n
        def f(idx):
            while not self.stop_event.is_set():
                try:
                    self.results[idx] = self.sample2batch(self.sample(mbsize))
                except Exception as e:
                    print("Exception while sampling:")
                    print(e)
                    self.sampler_threads[idx].failed = True
                    self.sampler_threads[idx].exception = e
                    self.ready_events[idx].set()
                    break
                self.ready_events[idx].set()
                self.resume_events[idx].clear()
                self.resume_events[idx].wait()
        self.sampler_threads = [threading.Thread(target=f, args=(i,)) for i in range(n)]
        [setattr(i, 'failed', False) for i in self.sampler_threads]
        [i.start() for i in self.sampler_threads]
        round_robin_idx = [0]
        def get():
            while True:
                idx = round_robin_idx[0]
                round_robin_idx[0] = (round_robin_idx[0] + 1) % n
                if self.ready_events[idx].is_set():
                    r = self.results[idx]
                    self.ready_events[idx].clear()
                    self.resume_events[idx].set()
                    return r
                elif round_robin_idx[0] == 0:
                    time.sleep(0.001)
        return get

    def stop_samplers_and_join(self):
        self.stop_event.set()
        if hasattr(self, 'sampler_threads'):
          while any([i.is_alive() for i in self.sampler_threads]):
            [i.set() for i in self.resume_events]
            [i.join(0.05) for i in self.sampler_threads]
    
    # def evaluate(self, epoch, algo = None):
    #     self.evaluator.add(self.sampled_mols[self.last_idx:])
    #     print('update size is ',len(self.sampled_mols)-self.last_idx)
    #     self.last_idx=len(self.sampled_mols)
    #     avg_topk_rs, avg_topk_tanimoto, num_modes_above_7_5, num_modes_above_8_0, \
    #         num_mols_above_7_5, num_mols_above_8_0= self.evaluator.eval_mols()
    #     print(f"state_visited={len(self.sampled_mols)};"
    #             f"num_modes R>7.5={num_modes_above_7_5};num_mols_above_7_5={num_mols_above_7_5};"
    #             f"num_modes R>8={num_modes_above_8_0};num_mols_above_8_0={num_mols_above_8_0};reward_top100:{avg_topk_rs[100]}")    
    #     self.oracle.update_history(num_modes_above_7_5)

    # tanimoto
    def evaluate(self, epoch, algo = None): 
        self.evaluator.add(self.sampled_mols[self.last_idx:])
        print('update size is ',len(self.sampled_mols)-self.last_idx)
        self.last_idx=len(self.sampled_mols)
        avg_topk_rs, avg_topk_tanimoto, num_modes_above_7_5, num_modes_above_8_0, \
            num_mols_above_7_5, num_mols_above_8_0= self.evaluator.eval_mols()
        print(f"state_visited={len(self.sampled_mols)};"
                f"num_modes R>7.5={num_modes_above_7_5};num_mols_above_7_5={num_mols_above_7_5};"
                f"num_modes R>8={num_modes_above_8_0};num_mols_above_8_0={num_mols_above_8_0};reward_topk:{avg_topk_rs};avg_topk_tanimoto:{avg_topk_tanimoto}")
        


def make_model(args, mdp, out_per_mol=1):
    if args.repr_type == 'block_graph':
        model = model_block.GraphAgent(nemb=args.nemb,
                                       nvec=0,
                                       out_per_stem=mdp.num_blocks,
                                       out_per_mol=out_per_mol,
                                       num_conv_steps=args.num_conv_steps,
                                       mdp_cfg=mdp,
                                       version=args.model_version)
    elif args.repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(nhid=args.nemb,
                                     nvec=0,
                                     num_out_per_stem=mdp.num_blocks,
                                     num_out_per_mol=out_per_mol,
                                     num_conv_steps=args.num_conv_steps,
                                     version=args.model_version,
                                     do_nblocks=(hasattr(args,'include_nblocks')
                                                 and args.include_nblocks), dropout_rate=0.1)
    elif args.repr_type == 'morgan_fingerprint':
        raise ValueError('reimplement me')
        model = model_fingerprint.MFP_MLP(args.nemb, 3, mdp.num_blocks, 1)
    return model

def get_mol_path_graph(mol):
    #bpath = "data/blocks_fix_131.json"
    bpath = "data/blocks_PDB_105.json"
    mdp = MolMDPExtended(bpath)
    mdp.post_init(torch.device('cpu'), 'block_graph')
    mdp.build_translation_table()
    mdp.floatX = torch.float
    agraph = nx.DiGraph()
    agraph.add_node(0)
    ancestors = [mol]
    ancestor_graphs = []

    par = mdp.parents(mol)
    mstack = [i[0] for i in par]
    pstack = [[0, a] for i,a in par]
    while len(mstack):
        m = mstack.pop() #pop = last item is default index
        p, pa = pstack.pop()
        match = False
        mgraph = mdp.get_nx_graph(m)
        for ai, a in enumerate(ancestor_graphs):
            if mdp.graphs_are_isomorphic(mgraph, a):
                agraph.add_edge(p, ai+1, action=pa)
                match = True
                break
        if not match:
            agraph.add_edge(p, len(ancestors), action=pa) #I assume the original molecule = 0, 1st ancestor = 1st parent = 1
            ancestors.append(m) #so now len(ancestors) will be 2 --> and the next edge will be to the ancestor labelled 2
            ancestor_graphs.append(mgraph)
            if len(m.blocks):
                par = mdp.parents(m)
                mstack += [i[0] for i in par]
                pstack += [(len(ancestors)-1, i[1]) for i in par]

    for u, v in agraph.edges:
        c = mdp.add_block_to(ancestors[v], *agraph.edges[(u,v)]['action'])
        geq = mdp.graphs_are_isomorphic(mdp.get_nx_graph(c, true_block=True),
                                        mdp.get_nx_graph(ancestors[u], true_block=True))
        if not geq: # try to fix the action
            block, stem = agraph.edges[(u,v)]['action']
            for i in range(len(ancestors[v].stems)):
                c = mdp.add_block_to(ancestors[v], block, i)
                geq = mdp.graphs_are_isomorphic(mdp.get_nx_graph(c, true_block=True),
                                                mdp.get_nx_graph(ancestors[u], true_block=True))
                if geq:
                    agraph.edges[(u,v)]['action'] = (block, i)
                    break
        if not geq:
            raise ValueError('could not fix action')
    for u in agraph.nodes:
        agraph.nodes[u]['mol'] = ancestors[u]
    return agraph

def compute_correlation(model, mdp, args):
    device = torch.device('cuda')
    tf = lambda x: torch.tensor(x, device=device).to(args.floatX)
    tint = lambda x: torch.tensor(x, device=device).long()

    test_mols = pickle.load(gzip.open('data/some_mols_U_1k.pkl.gz'))
    logsoftmax = nn.LogSoftmax(0)
    logp = []
    reward = []
    numblocks = []
    for moli in (test_mols[:1000]):
        reward.append(np.log(moli[0]))
        try:
            agraph = get_mol_path_graph(moli[1])
        except:
            continue
        s = mdp.mols2batch([mdp.mol2repr(agraph.nodes[i]['mol']) for i in agraph.nodes])
        numblocks.append(len(moli[1].blocks))
        with torch.no_grad():
            stem_out_s, mol_out_s = model(s, None)  # get the mols_out_s for ALL molecules not just the end one.
        per_mol_out = []
        # Compute pi(a|s)
        for j in range(len(agraph.nodes)):
            a,b = s._slice_dict['stems'][j:j+2]

            stop_allowed = len(agraph.nodes[j]['mol'].blocks) >= args.min_blocks
            mp = logsoftmax(torch.cat([
                stem_out_s[a:b].reshape(-1),
                # If num_blocks < min_blocks, the model is not allowed to stop
                mol_out_s[j, :1] if stop_allowed else tf([-1000])]))
            per_mol_out.append((mp[:-1].reshape((-1, stem_out_s.shape[1])), mp[-1]))

        # When the model reaches 8 blocks, it is stopped automatically. If instead it stops before
        # that, we need to take into account the STOP action's logprob
        if len(moli[1].blocks) < 8:
            stem_out_last, mol_out_last = model(mdp.mols2batch([mdp.mol2repr(moli[1])]), None)
            mplast = logsoftmax(torch.cat([stem_out_last.reshape(-1), mol_out_last[0, :1]]))
            MSTOP = mplast[-1]

        # assign logprob to edges
        for u,v in agraph.edges:
            a = agraph.edges[u,v]['action']
            if a[0] == -1:
                agraph.edges[u,v]['logprob'] = per_mol_out[v][1]
            else:
                agraph.edges[u,v]['logprob'] = per_mol_out[v][0][a[1], a[0]]

        # propagate logprobs through the graph
        for n in list(nx.topological_sort(agraph))[::-1]: 
            for c in agraph.predecessors(n): 
                if len(moli[1].blocks) < 8 and c == 0:
                    agraph.nodes[c]['logprob'] = torch.logaddexp(
                        agraph.nodes[c].get('logprob', tf(-1000)),
                        agraph.edges[c, n]['logprob'] + agraph.nodes[n].get('logprob', 0) + MSTOP)
                else:
                    agraph.nodes[c]['logprob'] = torch.logaddexp(
                        agraph.nodes[c].get('logprob', tf(-1000)),
                        agraph.edges[c, n]['logprob'] + agraph.nodes[n].get('logprob',0))

        logp.append((moli, agraph.nodes[n]['logprob'].item())) #add the first item
    return logp


class Proxy:
    def __init__(self, args, bpath, device):
        eargs = pickle.load(gzip.open(f'{args.proxy_path}/info.pkl.gz'))['args']
        params = pickle.load(gzip.open(f'{args.proxy_path}/best_params.pkl.gz'))
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, eargs.repr_type)
        self.mdp.floatX = args.floatX
        self.proxy = make_model(eargs, self.mdp)
        for a,b in zip(self.proxy.parameters(), params):
            a.data = torch.tensor(b, dtype=self.mdp.floatX)
        self.proxy.to(device)

    def __call__(self, m):
        m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        return self.proxy(m, do_stems=False)[1].item()

_stop = [None]

def log_resources(step: int):
    process = psutil.Process(os.getpid())
    cpu_usage = psutil.cpu_percent()
    mem_gb = process.memory_info().rss / (1024**3)  
    
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device_idx)
        mem_allocated = torch.cuda.memory_allocated(device_idx) / (1024**2)
        mem_reserved = torch.cuda.memory_reserved(device_idx) / (1024**2)
    else:
        device_idx = None
        device_name = "CPU"
        mem_allocated, mem_reserved = 0, 0

    print(f"[Resource] Step {step}: "
          f"CPU {cpu_usage:.1f}% | "
          f"RAM {mem_gb:.2f} GB | "
          f"GPU {device_name} (id {device_idx}) "
          f"Allocated {mem_allocated:.0f} MB / Reserved {mem_reserved:.0f} MB",
          flush=True)
    
def train_model_with_proxy(args, model, proxy, dataset, num_steps=None, do_save=True):
    debug_no_threads = True
    device = torch.device('cuda')

    if num_steps is None:
        num_steps = args.num_iterations + 1

    tau = args.bootstrap_tau
    if args.bootstrap_tau > 0:
        target_model = deepcopy(model)

    if do_save:
        exp_dir = f'{args.save_path}/{args.array}_{args.run}/'
        os.makedirs(exp_dir, exist_ok=True)


    dataset.set_sampling_model(model, proxy, sample_prob=args.sample_prob)

    def save_stuff():
        corr_logp = compute_correlation(model, dataset.mdp, args)
        with open(f'{args.partial}_{args.objective}_corr_logp.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['logR(x)', 'logP(x)'])
            writer.writerows(corr_logp)
        pickle.dump([i.data.cpu().numpy() for i in model.parameters()],
                    gzip.open(f'{exp_dir}/params.pkl.gz', 'wb'))

        pickle.dump(dataset.sampled_mols,
                    gzip.open(f'{exp_dir}/sampled_mols.pkl.gz', 'wb'))

        pickle.dump({'train_losses': train_losses,
                     'test_losses': test_losses,
                     'test_infos': test_infos,
                     'time_start': time_start,
                     'time_now': time.time(),
                     'args': args,},
                    gzip.open(f'{exp_dir}/info.pkl.gz', 'wb'))

        pickle.dump(train_infos,
                    gzip.open(f'{exp_dir}/train_info.pkl.gz', 'wb'))


    opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay,
                           betas=(args.opt_beta, args.opt_beta2),
                           eps=args.opt_epsilon)

    tf = lambda x: torch.tensor(x, device=device).to(args.floatX)
    tint = lambda x: torch.tensor(x, device=device).long()
    
    mbsize = args.mbsize
    ar = torch.arange(mbsize)

    if not debug_no_threads:
        sampler = dataset.start_samplers(8, mbsize)

    last_losses = []

    def stop_everything():
        print('joining')
        dataset.stop_samplers_and_join()
    _stop[0] = stop_everything

    train_losses = []
    test_losses = []
    test_infos = []
    train_infos = []
    time_start = time.time()
    time_last_check = time.time()

    loginf = 1000 # to prevent nans
    log_reg_c = args.log_reg_c
    clip_loss = tf([args.clip_loss])
    balanced_loss = args.balanced_loss
    do_nblocks_reg = False
    max_blocks = args.max_blocks
    leaf_coef = args.leaf_coef

    m_root = BlockMoleculeDataExtended()
    root_node = TreeNode(m_root, False)
    state_node_map[root_node.mol.smiles] = root_node

    for i in range(num_steps):
        if not debug_no_threads:
            r = sampler()
            for thread in dataset.sampler_threads:
                if thread.failed:
                    stop_everything()
                    pdb.post_mortem(thread.exception.__traceback__)
                    return
            p, pb, a, r, s, d, mols = r
        else:
            p, pb, a, r, s, d, mols = dataset.sample2batch(dataset.sample(mbsize, i))
        # Since we sampled 'mbsize' trajectories, we're going to get
        # roughly mbsize * H (H is variable) transitions
        ntransitions = r.shape[0]
        
        # state outputs
        if tau > 0:
            with torch.no_grad():
                stem_out_s, mol_out_s = target_model(s, None)
        else:
            stem_out_s, mol_out_s = model(s, None)
        # parents of the state outputs
        stem_out_p, mol_out_p = model(p, None)
        # index parents by their corresponding actions
        qsa_p = model.index_output_by_action(p, stem_out_p, mol_out_p[:, 0], a)
        # then sum the parents' contribution, this is the inflow
        exp_inflow = (torch.zeros((ntransitions,), device=device, dtype=dataset.floatX)
                      .index_add_(0, pb, torch.exp(qsa_p))) # pb is the parents' batch index
        inflow = torch.log(exp_inflow + log_reg_c)
        # sum the state's Q(s,a), this is the outflow
        exp_outflow = model.sum_output(s, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
        # include reward and done multiplier, then take the log
        # we're guarenteed that r > 0 iff d = 1, so the log always works
        outflow_plus_r = torch.log(log_reg_c + r + exp_outflow * (1-d))
        if do_nblocks_reg:
            losses = _losses = ((inflow - outflow_plus_r) / (s.nblocks * max_blocks)).pow(2)
        else:
            losses = _losses = (inflow - outflow_plus_r).pow(2)
        if clip_loss > 0:
            ld = losses.detach()
            losses = losses / ld * torch.minimum(ld, clip_loss)

        term_loss = (losses * d).sum() / (d.sum() + 1e-20)
        flow_loss = (losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
        if balanced_loss:
            loss = term_loss * leaf_coef + flow_loss
        else:
            loss = losses.mean()
        opt.zero_grad()
        loss.backward(retain_graph=(not i % 50))

        _term_loss = (_losses * d).sum() / (d.sum() + 1e-20)
        _flow_loss = (_losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
        last_losses.append((loss.item(), term_loss.item(), flow_loss.item()))
        train_losses.append((loss.item(), _term_loss.item(), _flow_loss.item(),
                             term_loss.item(), flow_loss.item()))
        if not i % 50:
            train_infos.append((
                _term_loss.data.cpu().numpy(),
                _flow_loss.data.cpu().numpy(),
                exp_inflow.data.cpu().numpy(),
                exp_outflow.data.cpu().numpy(),
                r.data.cpu().numpy(),
                mols[1],
                [i.pow(2).sum().item() for i in model.parameters()],
                torch.autograd.grad(loss, qsa_p, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_s, retain_graph=True)[0].data.cpu().numpy(),
                torch.autograd.grad(loss, stem_out_p, retain_graph=True)[0].data.cpu().numpy(),
            ))
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(),
                                           args.clip_grad)
        opt.step()
        model.training_steps = i + 1
        if tau > 0:
            for _a,b in zip(model.parameters(), target_model.parameters()):
                b.data.mul_(1-tau).add_(tau*_a)


        if not i % 10:
            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            print(i, last_losses)
            print('time:', time.time() - time_last_check)
            time_last_check = time.time()
            last_losses = []

            if not i % 1000 and do_save:
                save_stuff()
        print_interval = 10
        reset_interval = 100
        if not i%print_interval:
            log_resources(i)

            dataset.evaluate(i)
        if not i%reset_interval and i:
            # if dataset.oracle.decide_change():
            dataset.oracle.update_arms()
            dataset.oracle.oracle()
    stop_everything()
    if do_save:
        save_stuff()
    return model


def main(args):
    pid = os.getpid()
    print(f"Current PID: {pid}")
    set_seed(args.seed)
    print(f'seed fixed to {args.seed}')
    bpath = "data/blocks_PDB_105.json"
    device = torch.device('cuda')

    if args.floatX == 'float32':
        args.floatX = torch.float
    else:
        args.floatX = torch.double
    dataset = Dataset(args, bpath, device, floatX=args.floatX)
    print(args)


    mdp = dataset.mdp

    model = make_model(args, mdp)
    model.to(args.floatX)
    model.to(device)

    proxy = Proxy(args, bpath, device)

    train_model_with_proxy(args, model, proxy, dataset, do_save=True)
    print('Done.')


try:
    from arrays import*
except:
    print("no arrays")

good_config = {
    'replay_mode': 'online',
    'sample_prob': 1,
    'mbsize': 4,
    'max_blocks': 8,
    'min_blocks': 2,
    # This repr actually is pretty stable
    'repr_type': 'block_graph',
    'model_version': 'v4',
    'nemb': 256,
    # at 30k iterations the models usually have "converged" in the
    # sense that the reward distribution doesn't get better, but the
    # generated molecules keep being unique, so making this higher
    # should simply provide more high-reward states.
    'num_iterations': 30000,

    'R_min': 0.1,
    'log_reg_c': (0.1/8)**4,
    # This is to make reward roughly between 0 and 1 (proxy outputs
    # between ~0 and 10, but very few are above 8).
    'reward_norm': 8,
    # you can play with this, higher is more risky but will give
    # higher rewards on average if it succeeds.
    'reward_exp': 10,
    'learning_rate': 5e-4,
    'num_conv_steps': 10, # More steps is better but more expensive
    # Too low and there is less diversity, too high and the
    # high-reward molecules become so rare the model doesn't learn
    # about them, 0.05 and 0.02 are sensible values
    'random_action_prob': 0.05,
    'opt_beta2': 0.999, # Optimization seems very sensitive to this,
                        # default value works fine
    'leaf_coef': 10, # Can be much bigger, not sure what the trade off
                     # is exactly though
    'include_nblocks': False,
}

if __name__ == '__main__':
  args = parser.parse_args()
  if 0:
    all_hps = eval(args.array)(args)
    for run in range(len(all_hps)):
      args.run = run
      hps = all_hps[run]
      for k,v in hps.items():
        setattr(args, k, v)
      exp_dir = f'{args.save_path}/{args.array}_{args.run}/'
      #if os.path.exists(exp_dir):
      #  continue
      print(hps)
      main(args)
  elif args.array:
    all_hps = eval(args.array)(args)

    if args.print_array_length:
      print(len(all_hps))
    else:
      hps = all_hps[args.run]
      print(hps)
      for k,v in hps.items():
        setattr(args, k, v)
    try:
        main(args)
    except KeyboardInterrupt as e:
        print("stopping for", e)
        _stop[0]()
        raise e
    except Exception as e:
        print("exception", e)
        _stop[0]()
        raise e
  else:
      try:
          main(args)
      except KeyboardInterrupt as e:
          print("stopping for", e)
          _stop[0]()
          raise e
      except Exception as e:
          print("exception", e)
          _stop[0]()
          raise e