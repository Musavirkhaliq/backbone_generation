# say this is our graph data in which we are giving who to friend of whom

# A = [
#   [0, 1, 0, 0, 0],  # Alice connected to Bob
#   [1, 0, 1, 0, 0],  # Bob connected to Alice, Charlie
#   [0, 1, 0, 1, 0],  # Charlie connected to Bob, Dana
#   [0, 0, 1, 0, 1],  # Dana connected to Charlie, Eve
#   [0, 0, 0, 1, 0]   # Eve connected to Dana
# ]

# Now lets featurize each node i.e each person [likes_music, likes_sports, likes_movies]
# X = [
#   [1, 0, 1],  # Alice
#   [0, 1, 1],  # Bob
#   [0, 1, 0],  # Charlie
#   [1, 0, 0],  # Dana
#   [1, 1, 0]   # Eve
# ]

# Task: Friend Recommendation (Link Prediction)
# Goal: Predict missing edges (e.g., should Alice be friends with Charlie?).
# Training Data: Use existing edges as positive examples (1s) and sample some non-edges as negative examples (0s).
# Positive edges: (0, 1), (1, 2), (2, 3), (3, 4).
# Negative edges (sampled): (0, 2), (1, 4), (0, 4).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.data as data

# node features
X = torch.tensor([
    [1, 0, 1],  # Alice
    [0, 1, 1],  # Bob
    [0, 1, 0],  # Charlie
    [1, 0, 0],  # Dana
    [1, 1, 0]   # Eve
], dtype=torch.float)

# edge index (list of [source_node, target_node] pairs 2 * num_edges)
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4],  # Source nodes
    [1, 0, 2, 1, 3, 2, 4, 3]   # Target nodes (undirected, so both ways)
], dtype=torch.long)

train_edges = torch.tensor([
    [0, 1],  # Positive
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 2],  # Negative
    [1, 4],
    [0, 4]
],dtype=torch.long)
train_labels = torch.tensor([1, 1, 1, 1, 0, 0, 0], dtype=torch.float)

# Create a PyG Data object
data = data.Data(x=X, edge_index=edge_index)
print("edge_index",data.edge_index)

class GNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(GNN,self).__init__()
        self.conv1 = GCNConv(input_dim,hidden_dim)
        self.conv2 = GCNConv(hidden_dim,output_dim)

    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        # First layer aggregation + update
        x = self.conv1(x,edge_index)
        x = F.relu(x)
        # Second layer aggregation + update
        x = self.conv2(x,edge_index)
        print(x)
        return x
    
# Step 3: Link prediction function
def predict_edges(embeddings, edges):
    # Dot product between node pairs
    scores = (embeddings[edges[:, 0]] * embeddings[edges[:, 1]]).sum(dim=1)
    return torch.sigmoid(scores)

model = GNN(input_dim=3, hidden_dim=16, output_dim=2)

optimizer =torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

for epochs in range(10):
    optimizer.zero_grad()
    embeddings = model(data)
    pred = predict_edges(embeddings, train_edges)
    loss = F.binary_cross_entropy(pred, train_labels)
    loss.backward()
    optimizer.step()
    if True: #epochs % 10== 0:
        print(f"Epoch {epochs}, Loss: {loss.item()}")


model.eval()
with torch.no_grad():
    embeddings = model(data)
    all_pairs = torch.tensor([
        [i, j] for i in range(5) for j in range(i + 1, 5) if not (i,j) in [(0,1),(1,2),(2,3),(3,4)]
    ], dtype=torch.long)

    pred = predict_edges(embeddings, all_pairs)
    for pair, prob in zip(all_pairs, pred):
        print(f"User {pair[0]} - User {pair[1]}: {prob.item():.4f}")