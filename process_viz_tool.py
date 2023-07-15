import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import psutil
import random
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to fetch running processes and their key metrics (CPU usage, memory usage, etc.)
def get_running_processes():
    processes = []
    for process in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
        processes.append({
            'name': process.info['name'],
            'cpuUsage': process.info['cpu_percent'],
            'memoryUsage': process.info['memory_percent'],
            'numCores': psutil.cpu_count(logical=False)
        })
    # Sort the processes based on CPU usage (descending)
    processes.sort(key=lambda x: x['cpuUsage'], reverse=True)
    # Get the top 5 processes with the highest CPU usage
    top_5_processes = processes[:5]
    return top_5_processes

# Function to fetch subprocesses of a given main process
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
    # Limit to a maximum of 3 subprocesses per main process
    return subprocesses[:3]

# Function to update the network graph with the dynamic process hierarchy
def update_network_graph():
    try:
        top_5_processes = get_running_processes()

        G = nx.DiGraph()

        # Add tiers (main processes) with key metrics
        for process in top_5_processes:
            G.add_node(process['name'], node_type='tier', state='R', cpuUsage=process['cpuUsage'], memoryUsage=0, numCores=0)

        # Add nodes (subprocesses) with key metrics
        for process in top_5_processes:
            subprocesses = get_subprocesses(process['name'])
            for sub_process in subprocesses:
                G.add_node(sub_process['name'], node_type='node', state='G', cpuUsage=sub_process['cpuUsage'], memoryUsage=0, numCores=0)

        # Add edges (connections between tiers and nodes)
        for process in top_5_processes:
            subprocesses = get_subprocesses(process['name'])
            for sub_process in subprocesses:
                G.add_edge(process['name'], sub_process['name'], weight=1)

        # Clear the previous content of the figure
        plt.clf()

        # Draw the network graph
        pos = nx.spring_layout(G)

        # Create concentric circles for each node to represent health and metrics
        node_labels = {}
        node_colors = []
        node_sizes = []
        for node in G.nodes:
            node_data = G.nodes[node]
            if G.nodes[node].get('node_type', '') == 'tier':
                color = 'red' if node_data.get('state', '') == 'R' else 'yellow' if node_data.get('state', '') == 'Y' else 'green'
                node_colors.append(color)
                node_sizes.append(5000)
                label = f"{node}\nCPU: {node_data.get('cpuUsage', 0):.2f}%"
                node_labels[node] = label
            else:
                color = 'red' if node_data.get('state', '') == 'R' else 'yellow' if node_data.get('state', '') == 'Y' else 'green'
                node_colors.append(color)
                node_sizes.append(3000)
                label = f"{node}\nCPU: {node_data.get('cpuUsage', 0):.2f}%"
                node_labels[node] = label

        # Draw nodes and edges
        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes, font_size=10, labels=None, font_color='black', arrows=False, width=1.5, alpha=0.7)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_color='white')

        # Refresh the canvas
        canvas.draw()

    except Exception as e:
        # Log the exception with traceback and line number
        logging.exception(f"Error in update_network_graph: {e}")

# Function to update the GUI
def update_gui():
    update_network_graph()
    root.after(2000, update_gui)  # Refresh every 2 seconds

# Create the Tkinter GUI
root = tk.Tk()
root.title("AppDynamics-like Dynamic Network Visualization")

# Create a canvas to display the network graph
figure = plt.figure(figsize=(12, 8))
canvas = FigureCanvasTkAgg(figure, master=root)
canvas.get_tk_widget().pack()

# Start the GUI update loop
update_gui()

root.mainloop()
