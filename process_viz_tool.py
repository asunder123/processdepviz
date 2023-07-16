import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import networkx as nx
import psutil
import random
import logging
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

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


def update_network_graph(frame):
    try:
        top_5_processes = get_running_processes()

        G = nx.DiGraph()

        for process in top_5_processes:
            G.add_node(process['name'], node_type='tier', state='R', cpuUsage=process['cpuUsage'], memoryUsage=0, numCores=0)

        for process in top_5_processes:
            subprocesses = get_subprocesses(process['name'])
            for sub_process in subprocesses:
                G.add_node(sub_process['name'], node_type='node', state='G', cpuUsage=sub_process['cpuUsage'], memoryUsage=0, numCores=0)

        for process in top_5_processes:
            subprocesses = get_subprocesses(process['name'])
            for sub_process in subprocesses:
                G.add_edge(process['name'], sub_process['name'], weight=1)

        plt.clf()
        pos = nx.spring_layout(G)

        node_colors = []
        node_sizes = []
        for node in G.nodes:
            node_data = G.nodes[node]
            if G.nodes[node].get('node_type', '') == 'tier':
                color = get_color(node_data.get('cpuUsage', 0))
                node_colors.append(color)
                node_sizes.append(5000)
            else:
                color = get_color(node_data.get('cpuUsage', 0))
                node_colors.append(color)
                node_sizes.append(3000)

        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=10, labels=None, font_color='black', arrows=False, width=1.5, alpha=0.7)

        live_process_edges = []
        for process in psutil.process_iter(['name']):
            for edge in G.edges:
                if process.info['name'] in edge:
                    live_process_edges.append(edge)
        nx.draw_networkx_edges(G, pos, edgelist=live_process_edges, edge_color='blue', width=2.0)

    except Exception as e:
        logging.exception(f"Error in update_network_graph: {e}")


def get_color(cpu_usage):
    # Define a colormap based on CPU usage
    cmap = plt.get_cmap('coolwarm')  # Choose a colormap (coolwarm for blue-red)
    norm = mcolors.Normalize(vmin=0, vmax=100)  # Normalize CPU usage values to [0, 100]

    # Convert CPU usage to a color from the colormap
    color = cmap(norm(cpu_usage))
    return color


def update_gui():
    update_network_graph(0)
    canvas.draw()


root = tk.Tk()
root.title("AppDynamics-like Dynamic Network Visualization")

figure = plt.figure(figsize=(12, 8))
canvas = FigureCanvasTkAgg(figure, master=root)
canvas.get_tk_widget().pack()

ani = FuncAnimation(figure, update_network_graph, interval=2000)  # Update every 2 seconds
ani._start()

root.mainloop()
