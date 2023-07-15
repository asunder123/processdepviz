import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import psutil
import random

# Function to fetch running service names and their CPU usage
def get_running_services():
    processes = []
    for process in psutil.process_iter(['name', 'cpu_percent']):
        processes.append({
            'name': process.info['name'],
            'cpuUsage': process.info['cpu_percent']
        })
    # Sort the processes based on CPU usage (descending)
    processes.sort(key=lambda x: x['cpuUsage'], reverse=True)
    # Get the top 10 processes with the highest CPU usage
    top_10_processes = processes[:10]
    return top_10_processes

# Function to update the network graph with the dynamic service names and states
def update_network_graph():
    top_10_processes = get_running_services()

    # Create a simple network graph for demonstration with fetched service names and states
    dependency_data = {
        'nodes': [{'id': process['name'], 'state': 'R' if process['cpuUsage'] > 75 else 'G' if process['cpuUsage'] > 25 else 'B'}
                  for process in top_10_processes],
        'links': [
            {'source': top_10_processes[i]['name'], 'target': top_10_processes[i+1]['name'], 'value': 1}
            for i in range(len(top_10_processes) - 1)
        ]
    }

    G = nx.Graph()
    for node in dependency_data['nodes']:
        G.add_node(node['id'], state=node['state'])
    for link in dependency_data['links']:
        G.add_edge(link['source'], link['target'], weight=link['value'])

    # Clear the previous content of the figure
    plt.clf()

    # Draw the network graph
    pos = nx.spring_layout(G)
    node_colors = [node[1]['state'] for node in G.nodes(data=True)]
    # Map the 'R', 'G', 'B' states to corresponding colors
    node_colors = ['red' if color == 'R' else 'green' if color == 'G' else 'blue' for color in node_colors]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=1000, font_size=12)

    # Refresh the canvas
    canvas.draw()

# Function to update the GUI
def update_gui():
    update_network_graph()
    root.after(2000, update_gui)  # Refresh every 2 seconds

# Create the Tkinter GUI
root = tk.Tk()
root.title("Dynamic Network Visualization")

# Create a canvas to display the network graph
figure = plt.figure(figsize=(8, 6))
canvas = FigureCanvasTkAgg(figure, master=root)
canvas.get_tk_widget().pack()

# Start the GUI update loop
update_gui()

root.mainloop()
