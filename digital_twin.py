import pickle
import psutil
import random
import tkinter as tk
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import matplotlib.colors as mcolors

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the prediction model from the saved file
def load_prediction_model():
    with open('prediction_model.pkl', 'rb') as file:
        clf = pickle.load(file)
    return clf

# Function to generate a digital twin of a process based on CPU, RAM, and core information
def generate_digital_twin(process_name, cpu_usage, memory_usage, num_cores, prediction_model):
    # Simulate data collection and transformation (replace this with actual data collection)
    # In a real-world scenario, you would monitor the actual process and get its current CPU, RAM, and core usage
    # Here, we just use the provided CPU, RAM, and core values as an example
    process_data = [cpu_usage, memory_usage, num_cores]

    # Use the loaded prediction model to predict the process's failure status
    failure_prediction = prediction_model.predict([process_data])[0]

    # Simulate other digital twin properties or behaviors based on the failure_prediction

    # For example, you can print whether the process is predicted to fail or not
    if failure_prediction == 0:
        print(f"Process '{process_name}' is predicted to be stable (not fail).")
    else:
        print(f"Process '{process_name}' is predicted to be unstable (may fail).")

    # Other digital twin simulation tasks can be added based on the prediction and process data


def update_network_graph():
    try:
        prediction_model = load_prediction_model()

        G = nx.DiGraph()

        # Simulate Windows processes (replace this with actual process monitoring)
        for i in range(5):
            process_name = f"Process-{i}"
            cpu_usage = random.uniform(0, 100)
            memory_usage = random.uniform(0, 100)
            num_cores = psutil.cpu_count(logical=False)

            generate_digital_twin(process_name, cpu_usage, memory_usage, num_cores, prediction_model)

            G.add_node(process_name, node_type='process', cpuUsage=cpu_usage, memoryUsage=memory_usage, numCores=num_cores)

        plt.clf()
        pos = nx.spring_layout(G)

        node_colors = []
        node_sizes = []
        node_labels = {}
        for node in G.nodes:
            node_data = G.nodes[node]
            color = get_color(node_data.get('cpuUsage', 0))
            node_colors.append(color)
            node_sizes.append(5000)
            label = f"{node}\nCPU: {node_data.get('cpuUsage', 0):.2f}%\nRAM: {node_data.get('memoryUsage', 0):.2f}%\nCores: {node_data.get('numCores', 0)}"
            node_labels[node] = label

        nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=node_sizes, font_size=10, labels=None, font_color='black', arrows=False, width=1.5, alpha=0.7)

        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6, font_color='black')

        canvas.draw()
        root.after(2000, update_network_graph)  # Refresh every 2 seconds

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
