import random
import matplotlib.pyplot as plt
import numpy as np

# ---------------- CONFIG ----------------
CONFIG = {
    "time_steps": 200,        # number of time steps to simulate
    "base_demand": 50,        # average workload demand
    "demand_fluctuation": 30, # max random fluctuation
    "vm_capacity": 10,        # capacity handled by one VM
    "vm_cost": 2,             # cost per VM per time step
    "sla_threshold": 0.9,     # required service level (capacity >= demand)
    "reactive_delay": 2,      # delay in reacting to demand (for reactive strategy)
    "predictive_window": 5,   # steps used for demand prediction
    "random_seed": 42
}

random.seed(CONFIG["random_seed"])

# ---------------- DEMAND GENERATOR ----------------
def generate_demand(config):
    base = config["base_demand"]
    fluct = config["demand_fluctuation"]
    demand = []
    for t in range(config["time_steps"]):
        noise = random.uniform(-fluct, fluct)
        demand.append(max(0, base + noise))
    return demand

# ---------------- STRATEGIES ----------------
def static_strategy(config, demand):
    vm_needed = int(np.ceil(np.mean(demand) / config["vm_capacity"]))
    capacity = [vm_needed * config["vm_capacity"]] * config["time_steps"]
    return capacity

def reactive_strategy(config, demand):
    capacity = [0] * config["time_steps"]
    vms = 0
    for t in range(config["time_steps"]):
        if t >= config["reactive_delay"]:
            needed = int(np.ceil(demand[t - config["reactive_delay"]] / config["vm_capacity"]))
            vms = needed
        capacity[t] = vms * config["vm_capacity"]
    return capacity

def predictive_strategy(config, demand):
    capacity = [0] * config["time_steps"]
    for t in range(config["time_steps"]):
        window = demand[max(0, t - config["predictive_window"]):t + 1]
        predicted = np.mean(window) if window else config["base_demand"]
        vms = int(np.ceil(predicted / config["vm_capacity"]))
        capacity[t] = vms * config["vm_capacity"]
    return capacity

# ---------------- METRICS ----------------
def evaluate(config, demand, capacity):
    utilization = []
    sla_violations = 0
    total_cost = 0

    for d, c in zip(demand, capacity):
        utilization.append(min(d / c, 1) if c > 0 else 0)
        if c < d:
            sla_violations += 1
        total_cost += (c / config["vm_capacity"]) * config["vm_cost"]

    avg_utilization = np.mean(utilization)
    sla_viol_rate = sla_violations / config["time_steps"]

    return {
        "cost": total_cost,
        "utilization": avg_utilization,
        "sla_violations": sla_viol_rate
    }

# ---------------- MAIN SIMULATION ----------------
def simulate_and_plot():
    cfg = CONFIG
    demand = generate_demand(cfg)

    strategies = {
        "Static": static_strategy(cfg, demand),
        "Reactive": reactive_strategy(cfg, demand),
        "Predictive": predictive_strategy(cfg, demand),
    }

    results = {}
    for name, capacity in strategies.items():
        results[name] = evaluate(cfg, demand, capacity)

    # --- Plot demand vs capacity ---
    plt.figure(figsize=(10, 5))
    plt.plot(demand, label="Demand", color="black", linewidth=2)
    for name, capacity in strategies.items():
        plt.plot(capacity, label=f"{name} Capacity")
    plt.xlabel("Time Steps")
    plt.ylabel("Workload / Capacity")
    plt.title("Cloud Capacity vs Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig("resource_usage.png")
    plt.show()

    # --- Plot comparison metrics ---
    metrics = ["cost", "utilization", "sla_violations"]
    x = np.arange(len(metrics))
    width = 0.25
    plt.figure(figsize=(8, 4))
    for i, (name, res) in enumerate(results.items()):
        plt.bar(x + i * width, [res[m] for m in metrics], width, label=name)
    plt.xticks(x + width, metrics)
    plt.title("Comparison of Strategies")
    plt.legend()
    plt.tight_layout()
    plt.savefig("metrics_comparison.png")
    plt.show()

    # --- Print summary ---
    print("\n--- Simulation Summary ---")
    for name, res in results.items():
        print(f"{name}: Cost={res['cost']:.2f}, Utilization={res['utilization']:.2f}, SLA Violations={res['sla_violations']:.2%}")

if __name__ == "__main__":
    simulate_and_plot()
