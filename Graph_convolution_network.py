# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 20:52:01 2024

@author: horik
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, SAGEConv, AttentiveFP
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class AttentiveConvMoleculeModel(nn.Module):
    def __init__(self, num_node_features):
        super(AttentiveConvMoleculeModel, self).__init__()
        self.conv1 = GATConv(num_node_features, 64, heads=4, concat=False, edge_dim=10)
        self.conv2 = GATConv(64,64, heads=2,concat=False, edge_dim=10)
        self.conv3 = GATConv(64,64, heads=4, concat=False, edge_dim=10)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        
        x = global_mean_pool(x, batch)  # Global mean pooling
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.view(-1)
    
class ImprovedGCNMoleculeModel(nn.Module):
    def __init__(self, num_node_features):
        super(ImprovedGCNMoleculeModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GATConv(128, 128, heads=4, concat=False, edge_dim=10)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        
        x = global_mean_pool(x, batch)  # Global mean pooling
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.view(-1)
    
class MoleculeDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        smiles_list = df["SMILES"].values
        values = df["logS"].values
        data_list = [
            (self.smiles_to_graph(smiles), torch.tensor([label], dtype=torch.float))
            for smiles, label in zip(smiles_list, values)
        ]
        self.data_list = data_list

    @staticmethod
    def smiles_to_graph(smiles):
        mol = Chem.MolFromSmiles(smiles)
        symbols = ['C','N','O','Cl','S','Br','I','F','P']

        hybridizations = [Chem.rdchem.HybridizationType.S,
                          Chem.rdchem.HybridizationType.SP,
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3,
                          Chem.rdchem.HybridizationType.SP3D,
                          Chem.rdchem.HybridizationType.SP3D2,
                          'other',
                          ]

        stereos = [Chem.rdchem.BondStereo.STEREONONE,
                   Chem.rdchem.BondStereo.STEREOANY,
                   Chem.rdchem.BondStereo.STEREOZ,
                   Chem.rdchem.BondStereo.STEREOE,
                   ]
        
        node_features = []
        xs = []
        
        for atom in mol.GetAtoms():
            symbol = [0.] * len(symbols)
            symbol[symbols.index(atom.GetSymbol())] = 1.
            #comment degree from 6 to 8
            degree = [0.] * 8
            degree[atom.GetDegree()] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(hybridizations)
            hybridization[hybridizations.index(
                atom.GetHybridization())] = 1.
            aromaticity = 1. if atom.GetIsAromatic() else 0.
            hydrogens = [0.] * 5
            hydrogens[atom.GetTotalNumHs()] = 1.
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
        
            x = torch.tensor(symbol + degree + [formal_charge] +
                             [radical_electrons] + hybridization +
                             [aromaticity] + hydrogens + [chirality] +
                             chirality_type)
            xs.append(x)
        
            node_features = torch.stack(xs, dim=0)
    

        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
            edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]
            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
            stereo = [0.] * 4
            stereo[stereos.index(bond.GetStereo())] = 1.
    
            edge_attr = torch.tensor([single, double, triple, aromatic, conjugation, ring] + stereo)
    
            edge_attrs += [edge_attr, edge_attr]

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attrs, dim=0)

        return Data(x=x,edge_index=edge_index,edge_attr=edge_attr)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
dataset = MoleculeDataset(csv_path='logSdataset1290.csv')
test_size = 0.2
val_size = 0.2
train_val_data, test_data = train_test_split(dataset, test_size=test_size,random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=val_size/(1-test_size))
batch_size = 8
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

loaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    total_loss = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            loss = F.mse_loss(out, target.squeeze())
            total_loss += loss.item() * data.num_graphs
            predictions.append(out.cpu())
            targets.append(target.cpu())
    
    predictions = torch.cat(predictions, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    avg_loss = total_loss / len(loader.dataset)
    return predictions, targets, avg_loss

# モデルの初期化
model = AttentiveConvMoleculeModel(num_node_features=35)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss() 
# Early stoppingの設定
best_val_loss = float('inf')
patience = 10
counter = 0
num_epochs = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loss_history = []
val_loss_history = []
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, device)
    _, _, val_loss = evaluate(model, val_loader, device)
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    print(f'Epoch: {epoch+1:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')


# model.load_state_dict(torch.load('best_model.pth'))
_, _, test_loss = evaluate(model, test_loader, device)
print(f'Test Loss: {test_loss:.4f}')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(train_loss_history, label="train")
ax.plot(val_loss_history, label="val")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()

train_preds, train_targets, _ = evaluate(model, train_loader, device)
test_preds, test_targets, _ = evaluate(model, test_loader, device)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(train_targets, train_preds, color="tab:gray", alpha=0.5, label="train", s=20)
ax.scatter(test_targets, test_preds, color="tab:blue", label="test", s=20)
ax.plot([-12,2], [-12,2], color="gray", linewidth=1)
ax.set_xlim(-12, 2)
ax.set_ylim(-12, 2)
ax.set_aspect("equal")
ax.legend()
ax.set_xlabel("logS (true)")
ax.set_ylabel("logS (predict)")

train_r2 = r2_score(train_targets[:,0],train_preds)
test_r2 = r2_score(test_targets[:,0],test_preds)

print('train',round(train_r2,2), 'test',round(test_r2,2))
