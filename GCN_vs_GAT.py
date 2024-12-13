import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

###################################
# Set up device and load dataset
###################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cora_dataset = Planetoid(root='data/Cora', name='Cora')
cora_data = cora_dataset[0].to(device)

###################################
# Define GCN model
###################################
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

###################################
# Define GAT model
###################################
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, _ = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, _ = self.conv2(x, edge_index, return_attention_weights=True)
        return x

###################################
# Training function
###################################
def train(model, data, lr=0.005, wd=5e-4, epochs=200):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        train_acc = int((pred[data.train_mask] == data.y[data.train_mask]).sum()) / int(data.train_mask.sum())
        val_acc = int((pred[data.val_mask] == data.y[data.val_mask]).sum()) / int(data.val_mask.sum())
        test_acc = int((pred[data.test_mask] == data.y[data.test_mask]).sum()) / int(data.test_mask.sum())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

    return model, best_val_acc, best_test_acc

###################################
# Initialize and train models
###################################
model_gcn_cora = GCN(in_channels=cora_dataset.num_node_features,
                     hidden_channels=16,
                     out_channels=cora_dataset.num_classes)

model_gat_cora = GAT(in_channels=cora_dataset.num_node_features,
                     hidden_channels=8,
                     out_channels=cora_dataset.num_classes,
                     heads=8)

model_gcn_cora, gcn_val_acc_cora, gcn_test_acc_cora = train(model_gcn_cora, cora_data)
model_gat_cora, gat_val_acc_cora, gat_test_acc_cora = train(model_gat_cora, cora_data)

print("Cora - GCN: Val Acc:", gcn_val_acc_cora, "Test Acc:", gcn_test_acc_cora)
print("Cora - GAT: Val Acc:", gat_val_acc_cora, "Test Acc:", gat_test_acc_cora)

###################################
# Extract embeddings and visualize
###################################
model_gcn_cora.eval()
model_gat_cora.eval()

with torch.no_grad():
    embeddings_gcn = model_gcn_cora(cora_data.x, cora_data.edge_index).cpu().numpy()
    embeddings_gat = model_gat_cora(cora_data.x, cora_data.edge_index).cpu().numpy()

labels = cora_data.y.cpu().numpy()

tsne = TSNE(n_components=2, random_state=42)
emb_gcn_2d = tsne.fit_transform(embeddings_gcn)
emb_gat_2d = tsne.fit_transform(embeddings_gat)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(emb_gcn_2d[:,0], emb_gcn_2d[:,1], c=labels, cmap='tab10')
plt.title("GCN Embeddings")

plt.subplot(1,2,2)
plt.scatter(emb_gat_2d[:,0], emb_gat_2d[:,1], c=labels, cmap='tab10')
plt.title("GAT Embeddings")

plt.show()
