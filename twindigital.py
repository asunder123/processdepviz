# app.py

import psutil
import multiprocessing
from flask import Flask, jsonify, request

app = Flask(__name__)

# A list to store replicated processes and their metadata
replicated_processes = []

# Function to replicate the process

def replicate_process():
    # Add the functionality you want to replicate here
    while True:
        print("Child process: I am replicating the parent process!")
        # Simulate some work for the replicated process
        for _ in range(1000000):
            pass

# Route to get the list of running processes
@app.route("/processes", methods=["GET"])
def get_processes():
    process_list = []
    for process in psutil.process_iter(['pid', 'name']):
        process_list.append({
            "pid": process.info['pid'],
            "name": process.info['name']
        })
    return jsonify(process_list)

# Route to create a replicated process
@app.route("/replicate", methods=["POST"])
def create_replicated_process():
    process = multiprocessing.Process(target=replicate_process)
    process.start()

    # Store the Process object in the list for later reference
    replicated_processes.append({"process": process, "pid": process.pid, "running": True})

    return jsonify({"message": "Replicated process started.", "pid": process.pid})


# Route to get the list of replicated processes with metadata
@app.route("/replicated_processes", methods=["GET"])
def get_replicated_processes():
    replicated_process_list = []
    for process_info in replicated_processes:
        pid = process_info["pid"]
        try:
            process = psutil.Process(pid)
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=None)
            replicated_process_list.append({
                "pid": pid,
                "running": process_info["running"],
                "memory_usage": memory_info.rss,
                "cpu_percent": cpu_percent
            })
        except psutil.NoSuchProcess:
            # Handle the case when the process no longer exists
            process_info["running"] = False
            replicated_process_list.append({
                "pid": pid,
                "running": False,
                "memory_usage": None,
                "cpu_percent": None
            })
    # Remove the process from the list if it no longer exists
    replicated_processes[:] = [p for p in replicated_processes if p["running"]]
    return jsonify(replicated_process_list)

# Route to control the replicated processes
@app.route("/control_process", methods=["POST"])
def control_process():
    pid_to_control = request.json.get("pid")
    action = request.json.get("action")
    for process_info in replicated_processes:
        if process_info["pid"] == pid_to_control:
            process = process_info["process"]
            if action == "terminate":
                process.terminate()
                process_info["running"] = False
                return jsonify({"message": "Replicated process terminated.", "pid": pid_to_control})
            elif action == "pause":
                process.suspend()
                return jsonify({"message": "Replicated process paused.", "pid": pid_to_control})
            elif action == "resume":
                process.resume()
                return jsonify({"message": "Replicated process resumed.", "pid": pid_to_control})
    return jsonify({"error": "Process not found or invalid action."}), 404

if __name__ == "__main__":
    app.run(debug=True)
