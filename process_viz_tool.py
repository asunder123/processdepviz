import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import psutil
import random
import logging
import pickle  # To save the prediction model
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def get_running_processes():
    processes = []
    for process in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
        processes.append({
            'name': process.info['name'],
            'cpuUsage': process.info['cpu_percent'],
            'memoryUsage': process.info['memory_percent'],
            'numCores': psutil.cpu_count(logical=False)
        })
    processes.sort(key=lambda x: x['cpuUsage'], reverse=True)
    top_5_processes = processes[:5]
    return top_5_processes


def get_subprocesses(main_process_name):
    subprocesses = []
    for process in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
        if process.info['name'] == main_process_name:
            subprocesses.append({
                'name': f"Subprocess-{random.randint(1, 100)}",
                'cpuUsage': process.info['cpu_percent'],
                'memoryUsage': process.info['memory_percent'],
                'numCores': psutil.cpu_count(logical=False)
            })
    return subprocesses[:3]


def update_network_graph():
    try:
        top_5_processes = get_running_processes()

        G = nx.DiGraph()

        for process in top_5_processes:
            G.add_node(process['name'], node_type='tier', state='R', cpuUsage=process['cpuUsage'], memoryUsage=process['memoryUsage'], numCores=process['numCores'])

        for process in top_5_processes:
            subprocesses = get_subprocesses(process['name'])
            for sub_process in subprocesses:
                G.add_node(sub_process['name'], node_type='node', state='G', cpuUsage=sub_process['cpuUsage'], memoryUsage=sub_process['memoryUsage'], numCores=sub_process['numCores'])

        for process in top_5_processes:
            subprocesses = get_subprocesses(process['name'])
            for sub_process in subprocesses:
                call_rate = random.randint(1, 100)  # Random call rate for demonstration purposes
                G.add_edge(process['name'], sub_process['name'], weight=call_rate)

        plt.clf()
        pos = nx.spring_layout(G, seed=42)

        node_colors = []
        node_sizes = []
        node_labels = {}
        for node in G.nodes:
            node_data = G.nodes[node]
            if G.nodes[node].get('node_type', '') == 'tier':
                color = get_color(node_data.get('cpuUsage', 0))
                node_colors.append(color)
                node_sizes.append(5000)
                label = f"{node}\n{node_data.get('name')}\nCPU: {node_data.get('cpuUsage', 0):.2f}%\nRAM: {node_data.get('memoryUsage', 0):.2f}%\nCores: {node_data.get('numCores', 0)}"
                node_labels[node] = label
            else:
                color = get_color(node_data.get('cpuUsage', 0))
                node_colors.append(color)
                node_sizes.append(3000)
                label = f"{node}\n{node_data.get('name')}\nCPU: {node_data.get('cpuUsage', 0):.2f}%\nRAM: {node_data.get('memoryUsage', 0):.2f}%\nCores: {node_data.get('numCores', 0)}"
                node_labels[node] = label

        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_colors = ['blue' if weight > 50 else 'gray' for (_, _, weight) in G.edges.data('weight')]

        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes, font_size=10, labels=None, font_color='black', arrows=True, width=1.5, alpha=0.7, edge_color=edge_colors)

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='blue')

        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6, font_color='black')

        canvas.draw()
        root.after(2000, update_network_graph)  # Refresh every 2 seconds

        # Update the prediction model with random data
        X = []
        y = []
        for node in G.nodes:
            node_data = G.nodes[node]
            if G.nodes[node].get('node_type', '') == 'node':
                X.append([node_data['cpuUsage'], node_data['memoryUsage'], node_data['numCores']])
                # Simulate random failure (0 for not failed, 1 for failed)
                y.append(random.choice([0, 1]))

        # Create and train the prediction model (Random Forest Classifier)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        # Evaluate the model
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Model Accuracy:", accuracy)

        # Save the model to a file in the same directory as the app
        with open('prediction_model.pkl', 'wb') as file:
            pickle.dump(clf, file)

    except Exception as e:
        logging.exception(f"Error in update_network_graph: {e}")


def get_color(cpu_usage):
    cmap = plt.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=0, vmax=100)
    color = cmap(norm(cpu_usage))
    return color


root = tk.Tk()
root.title("Dynamic Network Visualization")

figure = plt.figure(figsize=(12, 8))
canvas = FigureCanvasTkAgg(figure, master=root)
canvas.get_tk_widget().pack()

update_network_graph()

root.mainloop()
