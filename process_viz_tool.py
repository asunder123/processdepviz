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
            'memoryUsage': process.info['memory_percent']
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
                'memoryUsage': process.info['memory_percent']
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
            G.add_node(process['name'], node_type='tier', state='R' if process['cpuUsage'] > 75 else 'G' if process['cpuUsage'] > 25 else 'B',
                       cpuUsage=f"CPU: {process['cpuUsage']}%", memoryUsage=f"Memory: {process['memoryUsage']}%")

        # Add nodes (subprocesses) with key metrics
        for process in top_5_processes:
            subprocesses = get_subprocesses(process['name'])
            for sub_process in subprocesses:
                G.add_node(sub_process['name'], node_type='node', state='R' if sub_process['cpuUsage'] > 75 else 'G' if sub_process['cpuUsage'] > 25 else 'B',
                           cpuUsage=f"CPU: {sub_process['cpuUsage']}%", memoryUsage=f"Memory: {sub_process['memoryUsage']}%")

        # Add edges (connections between tiers and nodes)
        for process in top_5_processes:
            subprocesses = get_subprocesses(process['name'])
            for sub_process in subprocesses:
                G.add_edge(process['name'], sub_process['name'], weight=1)

        # Clear the previous content of the figure
        plt.clf()

        # Draw the network graph
        pos = nx.spring_layout(G)

        # Map state 'R', 'G', 'B' to corresponding colors
        color_map = {'R': 'red', 'G': 'green', 'B': 'blue'}
        node_colors = [color_map[G.nodes[node].get('state', 'B')] for node in G.nodes]
        node_labels = {node: G.nodes[node].get('cpuUsage', '') + '\n' + G.nodes[node].get('memoryUsage', '') for node in G.nodes() if G.nodes[node].get('node_type', '') == 'tier'}

        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=2000, font_size=12, labels=node_labels, font_color='white')

        # Refresh the canvas
        canvas.draw()

    except Exception as e:
        # Log the exception with traceback and line number
        logging.exception(f"Error in update_network_graph: {e}")

# Function to update the GUI
def update_gui():
    try:
        update_network_graph()
        root.after(2000, update_gui)  # Refresh every 2 seconds

    except Exception as e:
        # Log the exception with traceback and line number
        logging.exception(f"Error in update_gui: {e}")

# Create the Tkinter GUI
root = tk.Tk()
root.title("AppDynamics-like Dynamic Network Visualization")

# Create a canvas to display the network graph
figure = plt.figure(figsize=(10, 6))
canvas = FigureCanvasTkAgg(figure, master=root)
canvas.get_tk_widget().pack()

# Start the GUI update loop
update_gui()

root.mainloop()
