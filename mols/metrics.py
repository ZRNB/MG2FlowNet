import gzip
import pickle
import rdkit.DataStructs
from rdkit import Chem
import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold
import heapq

def get_tanimoto_pairwise(mols):
    fps = [Chem.RDKFingerprint(i.mol) for i in mols]
    pairwise_sim = []
    for i in range(len(mols)):
        pairwise_sim.extend(rdkit.DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
    return pairwise_sim


class NumModes:
    def __init__(self, reward_exp, reward_norm, reward_thr=8, tanimoto_thr=0.7):
        self.reward_exp = reward_exp
        self.reward_norm = reward_norm
        self.reward_thr = reward_thr
        self.tanimoto_thr = tanimoto_thr
        self.modes = []
        self.max_reward = -1000
    def __call__(self, batch):
        candidates = []
        for some in batch:
            reward, mol = some[0], some[1]
            reward = (reward ** (1/self.reward_exp)) * self.reward_norm
            if reward > self.max_reward: 
                self.max_reward = reward
            if reward > self.reward_thr:
                candidates.append(mol)
        if len(candidates) > 0:
            # add one mode if needed
            if len(self.modes)==0: 
                self.modes.append(Chem.RDKFingerprint(candidates[0].mol))
            for mol in candidates:
                fp = Chem.RDKFingerprint(mol.mol)
                sims = np.asarray(rdkit.DataStructs.BulkTanimotoSimilarity(fp, self.modes))
                if all(sims < self.tanimoto_thr):
                    self.modes.append(fp)
        return self.max_reward, len(self.modes)
    def add(self, mols):
        candidates = []
        reward, mol = mols[0], mols[1]
        if reward > self.max_reward: 
            self.max_reward = reward
        if reward > self.reward_thr:
            candidates.append(mol)
        if len(candidates) > 0:
            if len(self.modes)==0: 
                self.modes.append(Chem.RDKFingerprint(candidates[0].mol))
            for mol in candidates:
                fp = Chem.RDKFingerprint(mol.mol)
                sims = np.asarray(rdkit.DataStructs.BulkTanimotoSimilarity(fp, self.modes))
                if all(sims < self.tanimoto_thr):
                    self.modes.append(fp)
    def get_modes(self):
        return len(self.modes)


# class Top1000:
#     def __init__(self):
#         self.min_heap = []  # 最小堆，存储前100大的元素
#         self.size = 1000

#     def add(self, num):
#         if len(self.min_heap) < self.size:
#             heapq.heappush(self.min_heap, num)
#         else:
#             if num > self.min_heap[0]:
#                 heapq.heapreplace(self.min_heap, num)

#     def get_top_1000(self):
#         return sorted(self.min_heap, reverse=True)
class _HeapItem:
    """Wrapper class to ensure only the reward (r) is compared."""
    def __init__(self, r, m):
        self.r = r
        self.m = m

    def __lt__(self, other):
        return self.r < other.r  # Only compare `r`, ignore `m`

class Top1000:
    def __init__(self):
        self.min_heap = []  # Stores _HeapItem objects
        self.size = 1000

    def add(self, item):
        r, m = item
        heap_item = _HeapItem(r, m)  # Wrap in our custom class
        if len(self.min_heap) < self.size:
            heapq.heappush(self.min_heap, heap_item)
        else:
            if r > self.min_heap[0].r:  # Compare against the smallest `r` in heap
                heapq.heapreplace(self.min_heap, heap_item)

    def get_top_1000(self):
        # Return sorted list of (r, m) tuples, descending by `r`
        sorted_items = sorted(self.min_heap, key=lambda x: -x.r)
        return [(item.r, item.m) for item in sorted_items]


# def eval_mols(mols, reward_norm=8, reward_exp=10, algo="gfn"):
#     def r2r_back(r):
#         return r ** (1. / reward_exp) * reward_norm
    
#     numModes_above_7_5 = NumModes(reward_exp=reward_exp, reward_norm=reward_norm, reward_thr=7.5)
#     _, num_modes_above_7_5 = numModes_above_7_5(mols)
#     numModes_above_8_0 = NumModes(reward_exp=reward_exp, reward_norm=reward_norm, reward_thr=8.)
#     _, num_modes_above_8_0 = numModes_above_8_0(mols)

    

#     top_ks = [10, 100, 1000]
#     avg_topk_rs = {}
#     avg_topk_tanimoto = {}
#     mol_r_map = {}

#     for i in range(len(mols)):
#         if algo == 'gfn':
#             r, m, trajectory_stats, inflow = mols[i]
#         else:
#             r, m = mols[i]
#         r = r2r_back(r)
#         mol = Chem.MolFromSmiles(m.smiles)  # 使用你感兴趣的分子的SMILES表示
#         scaffold = MurckoScaffold.GetScaffoldForMol(mol)  # 获取Bemis-Murcko骨架
#         scaffold_smiles = Chem.MolToSmiles(scaffold)
#         mol_r_map[scaffold_smiles] = r
    
#     unique_rs = list(mol_r_map.values())
#     unique_rs = sorted(unique_rs, reverse=True)
#     unique_rs = np.array(unique_rs)
#     num_above_7_5 = np.sum(unique_rs > 7.5) # just a integer
#     num_above_8_0 = np.sum(unique_rs > 8.0)

#     sorted_mol_r_map = sorted(mol_r_map.items(), key=lambda kv: kv[1], reverse=True)
#     for top_k_idx, top_k in enumerate(top_ks):
#         avg_topk_rs[top_k] = np.mean(unique_rs[:top_k])
        
#         topk_mols = [mol for (mol, r) in sorted_mol_r_map[:top_k]]
#         avg_topk_tanimoto[top_k] = np.mean(get_tanimoto_pairwise(topk_mols))

#     return avg_topk_rs, avg_topk_tanimoto, num_modes_above_7_5, num_modes_above_8_0, num_above_7_5, num_above_8_0

class Evaluator:
    def __init__(self,reward_norm=8, reward_exp=10, algo="gfn"):
        self.numModes_above_7_5=NumModes(reward_exp=reward_exp, reward_norm=reward_norm, reward_thr=7.5)
        self.numModes_above_8_0=NumModes(reward_exp=reward_exp, reward_norm=reward_norm, reward_thr=8.)
        self.top_ks = [10, 100, 1000]
        self.avg_topk_tanimoto = {}
        self.mol_r_map = {}
        self.reward_norm=reward_norm
        self.reward_exp=reward_exp
        self.algo=algo
        self.map_cnt_7_5=0
        self.map_cnt_8_0=0
        self.top1000 = Top1000()

    def r2r_back(self,r):
        return r ** (1. / self.reward_exp) * self.reward_norm
    def add(self,batch):
        for i in range(len(batch)):
            if self.algo == 'gfn':
                r, m, trajectory_stats, inflow = batch[i]
            else:
                r, m = batch[i]
            r = self.r2r_back(r)
            mol = Chem.MolFromSmiles(m.smiles)  # 使用你感兴趣的分子的SMILES表示
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)  # 获取Bemis-Murcko骨架
            scaffold_smiles = Chem.MolToSmiles(scaffold)

            if scaffold_smiles not in self.mol_r_map or self.mol_r_map[scaffold_smiles]<r:
                self.mol_r_map[scaffold_smiles] = r
                if r>7.5:self.map_cnt_7_5+=1
                if r>8.0:self.map_cnt_8_0+=1
                self.top1000.add((r,m))
                
            self.numModes_above_7_5.add((r,m))
            self.numModes_above_8_0.add((r,m))
    
    def eval_mols(self):
        avg_topk_rs = {}
        avg_topk_tanimoto = {}
        top_lst = self.top1000.get_top_1000()
        for top_k_idx, top_k in enumerate(self.top_ks):
            num_values = [r for (r,mol) in top_lst[:top_k]]
            topk_mols = [mol for (r, mol) in top_lst[:top_k]]
            avg_topk_tanimoto[top_k] = np.mean(get_tanimoto_pairwise(topk_mols))
            avg_topk_rs[top_k] = np.mean(num_values[:top_k])
        num_modes_above_7_5= self.numModes_above_7_5.get_modes()
        num_modes_above_8_0= self.numModes_above_8_0.get_modes()
        num_above_7_5 = self.map_cnt_7_5
        num_above_8_0 = self.map_cnt_8_0

        return avg_topk_rs, avg_topk_tanimoto, num_modes_above_7_5, num_modes_above_8_0, num_above_7_5, num_above_8_0