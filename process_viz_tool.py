import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import MessagePassing, GCNConv
import psutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Prepare the dataset (using psutil to get CPU and memory usage)
def get_running_processes_data():
    processes_data = []
    for process in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        processes_data.append({
            'pid': process.info['pid'],
            'name': process.info['name'],
            'cpuUsage': process.info['cpu_percent'],
            'memoryUsage': process.info['memory_percent'],
        })
    return processes_data

processes_data = get_running_processes_data()

# Print the number of processes and some sample data
num_processes = len(processes_data)
print("Line 22: Number of processes:", num_processes)
print("Line 23: Sample process data:", processes_data[0])

# Prepare the feature matrix (X) and target variable (y)
X = [[process['cpuUsage'], process['memoryUsage']] for process in processes_data]
y = [1 if process['cpuUsage'] > 50 or process['memoryUsage'] > 50 else 0 for process in processes_data]

# Convert NumPy arrays to PyTorch tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# Step 2: Create edge index for a directed graph (process dependencies)
num_nodes = len(processes_data)
edge_index = []
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:
            edge_index.append([i, j])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Remove duplicates and self-loops to handle disconnected nodes
def remove_self_loops(edge_index):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    return edge_index

edge_index = remove_self_loops(edge_index)

# Handle out-of-bound indices in edge_index
num_nodes = X.size(0)
edge_index = edge_index[:, edge_index[0] < num_nodes]
edge_index = edge_index[:, edge_index[1] < num_nodes]

# Print the number of edges and some sample data
num_edges = edge_index.size(1)
print("Line 49: Number of edges:", num_edges)
print("Line 50: Sample edge data:", edge_index[:, 0])

# Step 3: Define the GNN Model
class MyGNN(MessagePassing):
    def __init__(self):
        super(MyGNN, self).__init__(aggr='add')
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, x, edge_index):
        # Precheck dimensions before convolution
        if x.size(0) < edge_index.max().item() + 1:
            x = F.pad(x, (0, 0, 0, edge_index.max().item() + 1 - x.size(0)), "constant", 0)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Step 4: Train the GNN Model
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a PyTorch DataLoader for handling batches
train_data = Data(x=X_train, edge_index=edge_index)

# Define the batch size
batch_size = 20



# Create the DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Step 5: Save the GNN Model
gnn_model = MyGNN()
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)

gnn_model.train()

data_iterator = iter(train_loader)

# Initialize variables for iterative batch size adjustment
batch_size = 10
max_batch_size = 200
saved_model = False

while not saved_model and batch_size <= max_batch_size:
    # Re-create DataLoader with updated batch size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(100):  # Train for 100 epochs (you can adjust this value)
        try:
            data = next(data_iterator)
        except StopIteration:
            # If the DataLoader reaches the end, reset the iterator
            data_iterator = iter(train_loader)
            data = next(data_iterator)

        if data.num_graphs == 0:
            print("Line 95: Empty graph encountered, skipping batch.")  # Debug statement
            continue  # Skip empty graphs

        optimizer.zero_grad()
        out = gnn_model(data.x, data.edge_index)

        # Gather the target labels for each node in the batch
        target_labels = y_train[data.ptr: data.ptr + data.num_graphs * data.num_nodes]

        try:
            loss = F.cross_entropy(out, target_labels)
        except IndexError:
            # If there is an IndexError, continue to the next batch
            continue

        loss.backward()
        optimizer.step()

        # Update the pointer to the next batch
        data.ptr += data.num_graphs * data.num_nodes

    try:
        # Attempt to save the model
        torch.save(gnn_model.state_dict(), 'gnn_model.pt')
        saved_model = True
    except Exception as e:
        print(f"Line 123: Error saving the model: {e}")
        # Increase the batch size and reset the model
        batch_size += 10
        gnn_model = MyGNN()
        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
        gnn_model.train()
        data_iterator = iter(train_loader)

# Check if the model was saved successfully
if saved_model:
    print(f"Line 134: Model saved successfully with batch size: {batch_size}")
else:
    print("Line 136: Model could not be saved even after adjusting the batch size.")

# Step 6: Load the saved GNN Model (optional - for demonstration purposes)
loaded_gnn_model = MyGNN()
loaded_gnn_model.load_state_dict(torch.load('gnn_model.pt'))
loaded_gnn_model.eval()

# Step 7: Use the GNN Model for Prediction
with torch.no_grad():
    test_data = Data(x=X_test, edge_index=edge_index)
    out = loaded_gnn_model(test_data.x, test_data.edge_index)
    predicted_labels = out.argmax(dim=1)

# Step 8: Evaluate the Model
accuracy = (predicted_labels == y_test).sum().item() / y_test.size(0)
print("Line 149: Test Accuracy:", accuracy)

# Step 9: Visualize the Process Dependencies (Graph Visualization)
# Create a networkx graph from the edge index
G = nx.DiGraph()
for edge in edge_index.t().tolist():
    G.add_edge(edge[0], edge[1])

# Draw the graph
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10)
plt.title("Process Dependencies Graph")
plt.show()

# Note: This code assumes a directed graph for process dependencies. You may need to create a different graph
# structure based on your specific use case.
