import json
import math
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Configuration
SNAPSHOT_DIR = Path("snapshots")
CLASSICAL_PATTERN = re.compile(r"cutin_classical_t([0-9.]+)\.json")
QUANTUM_PATTERN = re.compile(r"cutin_quantum_t([0-9.]+)\.json")


def load_snapshots(pattern):
    """Load all snapshot files matching the pattern"""
    snapshots = []

    if not SNAPSHOT_DIR.exists():
        print(f"Error: {SNAPSHOT_DIR} directory not found!")
        return snapshots

    for file in SNAPSHOT_DIR.glob("*.json"):
        match = pattern.match(file.name)
        if match:
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    snapshots.append(data)
            except Exception as e:
                print(f"Warning: Could not load {file.name}: {e}")

    # Sort by simulation time
    snapshots.sort(key=lambda x: x["sim_time"])
    return snapshots


def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two 3D positions"""
    return math.sqrt(
        (pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2
    )


def calculate_speed(velocity):
    """Calculate speed magnitude from velocity vector"""
    return math.sqrt(velocity[0] ** 2 + velocity[1] ** 2 + velocity[2] ** 2)


def calculate_relative_position(ego_loc, npc_loc, ego_vel):
    """Calculate longitudinal and lateral distance"""
    # Direction vector from ego velocity
    speed = calculate_speed(ego_vel)
    if speed < 0.1:
        # If ego is stationary, use simple difference
        return npc_loc[1] - ego_loc[1], npc_loc[0] - ego_loc[0]

    # Forward direction (normalized velocity)
    forward = [ego_vel[0] / speed, ego_vel[1] / speed, 0]
    # Right direction (perpendicular)
    right = [-forward[1], forward[0], 0]

    # Difference vector
    diff = [npc_loc[0] - ego_loc[0], npc_loc[1] - ego_loc[1], 0]

    # Project onto forward (longitudinal) and right (lateral)
    longitudinal = diff[0] * forward[0] + diff[1] * forward[1]
    lateral = diff[0] * right[0] + diff[1] * right[1]

    return longitudinal, lateral


def analyze_snapshots(snapshots, algorithm_name):
    """Extract key metrics from snapshots"""
    metrics = {
        "times": [],
        "distances": [],
        "longitudinal": [],
        "lateral": [],
        "ego_speeds": [],
        "npc_speeds": [],
        "relative_speeds": [],
        "min_distance": float("inf"),
        "min_distance_time": 0,
        "collision": False,
    }

    for snap in snapshots:
        time = snap["sim_time"]
        ego_loc = snap["ego"]["loc"]
        ego_vel = snap["ego"]["vel"]
        npc_loc = snap["npc"]["loc"]
        npc_vel = snap["npc"]["vel"]

        # Calculate metrics
        distance = calculate_distance(ego_loc, npc_loc)
        ego_speed = calculate_speed(ego_vel)
        npc_speed = calculate_speed(npc_vel)
        relative_speed = ego_speed - npc_speed
        longitudinal, lateral = calculate_relative_position(ego_loc, npc_loc, ego_vel)

        # Store metrics
        metrics["times"].append(time)
        metrics["distances"].append(distance)
        metrics["longitudinal"].append(longitudinal)
        metrics["lateral"].append(lateral)
        metrics["ego_speeds"].append(ego_speed)
        metrics["npc_speeds"].append(npc_speed)
        metrics["relative_speeds"].append(relative_speed)

        # Track minimum distance
        if distance < metrics["min_distance"]:
            metrics["min_distance"] = distance
            metrics["min_distance_time"] = time

        # Check for collision (distance < 2 meters)
        if distance < 2.0:
            metrics["collision"] = True

    return metrics


def calculate_statistics(metrics):
    """Calculate statistical measures"""
    stats = {
        "avg_distance": np.mean(metrics["distances"]) if metrics["distances"] else 0,
        "min_distance": metrics["min_distance"],
        "min_distance_time": metrics["min_distance_time"],
        "avg_ego_speed": np.mean(metrics["ego_speeds"]) if metrics["ego_speeds"] else 0,
        "avg_npc_speed": np.mean(metrics["npc_speeds"]) if metrics["npc_speeds"] else 0,
        "std_distance": np.std(metrics["distances"])
        if len(metrics["distances"]) > 1
        else 0,
        "collision": metrics["collision"],
        "time_below_10m": sum(1 for d in metrics["distances"] if d < 10)
        * 0.5,  # Assuming 0.5s intervals
        "time_below_5m": sum(1 for d in metrics["distances"] if d < 5) * 0.5,
    }
    return stats


def print_comparison_report(
    classical_metrics, quantum_metrics, classical_stats, quantum_stats
):
    """Print a detailed comparison report"""
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON REPORT")
    print("=" * 80)

    print("\n--- SAFETY METRICS ---")
    print(f"{'Metric':<30} {'Classical':<20} {'Quantum':<20} {'Winner':<10}")
    print("-" * 80)

    # Minimum distance (higher is better)
    c_min = classical_stats["min_distance"]
    q_min = quantum_stats["min_distance"]
    winner = "Quantum" if q_min > c_min else ("Classical" if c_min > q_min else "Tie")
    print(f"{'Min Distance (m)':<30} {c_min:<20.2f} {q_min:<20.2f} {winner:<10}")

    # Average distance (higher is better)
    c_avg = classical_stats["avg_distance"]
    q_avg = quantum_stats["avg_distance"]
    winner = "Quantum" if q_avg > c_avg else ("Classical" if c_avg > q_avg else "Tie")
    print(f"{'Avg Distance (m)':<30} {c_avg:<20.2f} {q_avg:<20.2f} {winner:<10}")

    # Collision status (False is better)
    c_col = classical_stats["collision"]
    q_col = quantum_stats["collision"]
    winner = (
        "Quantum"
        if not q_col and c_col
        else (
            "Classical"
            if not c_col and q_col
            else ("Both Safe" if not c_col and not q_col else "Both Collided")
        )
    )
    print(f"{'Collision Occurred':<30} {str(c_col):<20} {str(q_col):<20} {winner:<10}")

    # Time in danger zones (lower is better)
    c_time_10 = classical_stats["time_below_10m"]
    q_time_10 = quantum_stats["time_below_10m"]
    winner = (
        "Quantum"
        if q_time_10 < c_time_10
        else ("Classical" if c_time_10 < q_time_10 else "Tie")
    )
    print(f"{'Time < 10m (s)':<30} {c_time_10:<20.2f} {q_time_10:<20.2f} {winner:<10}")

    c_time_5 = classical_stats["time_below_5m"]
    q_time_5 = quantum_stats["time_below_5m"]
    winner = (
        "Quantum"
        if q_time_5 < c_time_5
        else ("Classical" if c_time_5 < q_time_5 else "Tie")
    )
    print(f"{'Time < 5m (s)':<30} {c_time_5:<20.2f} {q_time_5:<20.2f} {winner:<10}")

    print("\n--- PERFORMANCE METRICS ---")
    print(f"{'Metric':<30} {'Classical':<20} {'Quantum':<20}")
    print("-" * 80)
    print(
        f"{'Avg Ego Speed (m/s)':<30} {classical_stats['avg_ego_speed']:<20.2f} {quantum_stats['avg_ego_speed']:<20.2f}"
    )
    print(
        f"{'Avg NPC Speed (m/s)':<30} {classical_stats['avg_npc_speed']:<20.2f} {quantum_stats['avg_npc_speed']:<20.2f}"
    )
    print(
        f"{'Distance Std Dev (m)':<30} {classical_stats['std_distance']:<20.2f} {quantum_stats['std_distance']:<20.2f}"
    )

    print("\n--- KEY EVENTS ---")
    print(f"{'Event':<30} {'Classical':<20} {'Quantum':<20}")
    print("-" * 80)
    print(
        f"{'Min Distance Time (s)':<30} {classical_stats['min_distance_time']:<20.2f} {quantum_stats['min_distance_time']:<20.2f}"
    )

    # Calculate safety score (0-100)
    def calculate_safety_score(stats):
        score = 100
        if stats["collision"]:
            score = 0  # Auto-fail
        else:
            # Penalties
            if stats["min_distance"] < 3:
                score -= 50
            elif stats["min_distance"] < 5:
                score -= 30
            elif stats["min_distance"] < 7:
                score -= 15

            # Additional penalties
            score -= min(20, stats["time_below_5m"] * 5)  # 5 points per second below 5m
            score -= min(
                10, stats["time_below_10m"] * 2
            )  # 2 points per second below 10m

        return max(0, score)

    c_score = calculate_safety_score(classical_stats)
    q_score = calculate_safety_score(quantum_stats)

    print("\n--- OVERALL SAFETY SCORE (0-100) ---")
    print(f"Classical: {c_score:.1f}/100")
    print(f"Quantum:   {q_score:.1f}/100")
    print(
        f"Winner: {'Quantum' if q_score > c_score else ('Classical' if c_score > q_score else 'Tie')}"
    )

    print("\n" + "=" * 80)


def plot_comparison(classical_metrics, quantum_metrics):
    """Create visualization plots comparing both algorithms"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Classical vs Quantum Algorithm Comparison", fontsize=16, fontweight="bold"
    )

    # Plot 1: Distance over time
    ax1 = axes[0, 0]
    ax1.plot(
        classical_metrics["times"],
        classical_metrics["distances"],
        "b-",
        label="Classical",
        linewidth=2,
        alpha=0.8,
    )
    ax1.plot(
        quantum_metrics["times"],
        quantum_metrics["distances"],
        "r-",
        label="Quantum",
        linewidth=2,
        alpha=0.8,
    )
    ax1.axhline(
        y=5, color="orange", linestyle="--", label="Danger Zone (5m)", alpha=0.5
    )
    ax1.axhline(y=2, color="red", linestyle="--", label="Collision (2m)", alpha=0.5)
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Distance (m)", fontsize=12)
    ax1.set_title("Inter-Vehicle Distance Over Time", fontsize=14, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Longitudinal distance (ahead/behind)
    ax2 = axes[0, 1]
    ax2.plot(
        classical_metrics["times"],
        classical_metrics["longitudinal"],
        "b-",
        label="Classical",
        linewidth=2,
        alpha=0.8,
    )
    ax2.plot(
        quantum_metrics["times"],
        quantum_metrics["longitudinal"],
        "r-",
        label="Quantum",
        linewidth=2,
        alpha=0.8,
    )
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax2.set_xlabel("Time (s)", fontsize=12)
    ax2.set_ylabel("Longitudinal Distance (m)", fontsize=12)
    ax2.set_title(
        "Longitudinal Position (+ = NPC ahead)", fontsize=14, fontweight="bold"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Lateral distance (lane position)
    ax3 = axes[1, 0]
    ax3.plot(
        classical_metrics["times"],
        classical_metrics["lateral"],
        "b-",
        label="Classical",
        linewidth=2,
        alpha=0.8,
    )
    ax3.plot(
        quantum_metrics["times"],
        quantum_metrics["lateral"],
        "r-",
        label="Quantum",
        linewidth=2,
        alpha=0.8,
    )
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.3, label="Ego Lane Center")
    ax3.axhline(y=3.5, color="green", linestyle="--", alpha=0.3, label="Right Lane")
    ax3.set_xlabel("Time (s)", fontsize=12)
    ax3.set_ylabel("Lateral Distance (m)", fontsize=12)
    ax3.set_title("Lateral Position (+ = right of ego)", fontsize=14, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Speed comparison
    ax4 = axes[1, 1]
    ax4.plot(
        classical_metrics["times"],
        classical_metrics["ego_speeds"],
        "b-",
        label="Classical Ego",
        linewidth=2,
        alpha=0.8,
    )
    ax4.plot(
        quantum_metrics["times"],
        quantum_metrics["ego_speeds"],
        "r-",
        label="Quantum Ego",
        linewidth=2,
        alpha=0.8,
    )
    ax4.plot(
        classical_metrics["times"],
        classical_metrics["npc_speeds"],
        "b--",
        label="Classical NPC",
        linewidth=2,
        alpha=0.6,
    )
    ax4.plot(
        quantum_metrics["times"],
        quantum_metrics["npc_speeds"],
        "r--",
        label="Quantum NPC",
        linewidth=2,
        alpha=0.6,
    )
    ax4.set_xlabel("Time (s)", fontsize=12)
    ax4.set_ylabel("Speed (m/s)", fontsize=12)
    ax4.set_title("Vehicle Speeds Over Time", fontsize=14, fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_file = "algorithm_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n[INFO] Comparison plot saved to: {output_file}")

    plt.show()


def export_metrics_to_csv(
    classical_metrics, quantum_metrics, classical_stats, quantum_stats
):
    """Export detailed metrics to CSV for further analysis"""
    import csv

    # Time-series data
    with open("comparison_timeseries.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Time",
                "Classical_Distance",
                "Quantum_Distance",
                "Classical_Longitudinal",
                "Quantum_Longitudinal",
                "Classical_Lateral",
                "Quantum_Lateral",
                "Classical_Ego_Speed",
                "Quantum_Ego_Speed",
                "Classical_NPC_Speed",
                "Quantum_NPC_Speed",
            ]
        )

        for i in range(
            min(len(classical_metrics["times"]), len(quantum_metrics["times"]))
        ):
            writer.writerow(
                [
                    classical_metrics["times"][i],
                    classical_metrics["distances"][i],
                    quantum_metrics["distances"][i],
                    classical_metrics["longitudinal"][i],
                    quantum_metrics["longitudinal"][i],
                    classical_metrics["lateral"][i],
                    quantum_metrics["lateral"][i],
                    classical_metrics["ego_speeds"][i],
                    quantum_metrics["ego_speeds"][i],
                    classical_metrics["npc_speeds"][i],
                    quantum_metrics["npc_speeds"][i],
                ]
            )

    print(f"[INFO] Time-series data exported to: comparison_timeseries.csv")

    # Summary statistics
    with open("comparison_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Classical", "Quantum"])
        writer.writerow(
            [
                "Min Distance (m)",
                classical_stats["min_distance"],
                quantum_stats["min_distance"],
            ]
        )
        writer.writerow(
            [
                "Avg Distance (m)",
                classical_stats["avg_distance"],
                quantum_stats["avg_distance"],
            ]
        )
        writer.writerow(
            [
                "Std Distance (m)",
                classical_stats["std_distance"],
                quantum_stats["std_distance"],
            ]
        )
        writer.writerow(
            [
                "Min Distance Time (s)",
                classical_stats["min_distance_time"],
                quantum_stats["min_distance_time"],
            ]
        )
        writer.writerow(
            [
                "Avg Ego Speed (m/s)",
                classical_stats["avg_ego_speed"],
                quantum_stats["avg_ego_speed"],
            ]
        )
        writer.writerow(
            [
                "Avg NPC Speed (m/s)",
                classical_stats["avg_npc_speed"],
                quantum_stats["avg_npc_speed"],
            ]
        )
        writer.writerow(
            [
                "Time Below 10m (s)",
                classical_stats["time_below_10m"],
                quantum_stats["time_below_10m"],
            ]
        )
        writer.writerow(
            [
                "Time Below 5m (s)",
                classical_stats["time_below_5m"],
                quantum_stats["time_below_5m"],
            ]
        )
        writer.writerow(
            ["Collision", classical_stats["collision"], quantum_stats["collision"]]
        )

    print(f"[INFO] Summary statistics exported to: comparison_summary.csv")


def main():
    print("=" * 80)
    print("QUANTUM vs CLASSICAL ALGORITHM COMPARISON")
    print("=" * 80)
    print("\nLoading snapshot data...")

    # Load snapshots
    classical_snapshots = load_snapshots(CLASSICAL_PATTERN)
    quantum_snapshots = load_snapshots(QUANTUM_PATTERN)

    print(f"  Classical snapshots: {len(classical_snapshots)}")
    print(f"  Quantum snapshots:   {len(quantum_snapshots)}")

    if not classical_snapshots or not quantum_snapshots:
        print("\n[ERROR] Could not find snapshot files!")
        print(f"Make sure JSON files exist in: {SNAPSHOT_DIR.absolute()}")
        print(
            "Expected filenames: cutin_classical_tX.XX.json and cutin_quantum_tX.XX.json"
        )
        return

    # Analyze snapshots
    print("\nAnalyzing classical algorithm...")
    classical_metrics = analyze_snapshots(classical_snapshots, "classical")
    classical_stats = calculate_statistics(classical_metrics)

    print("Analyzing quantum algorithm...")
    quantum_metrics = analyze_snapshots(quantum_snapshots, "quantum")
    quantum_stats = calculate_statistics(quantum_metrics)

    # Print comparison report
    print_comparison_report(
        classical_metrics, quantum_metrics, classical_stats, quantum_stats
    )

    # Export data
    print("\n" + "=" * 80)
    print("EXPORTING DATA")
    print("=" * 80)
    export_metrics_to_csv(
        classical_metrics, quantum_metrics, classical_stats, quantum_stats
    )

    # Create plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    try:
        plot_comparison(classical_metrics, quantum_metrics)
    except Exception as e:
        print(f"[WARNING] Could not generate plots: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - algorithm_comparison.png (visualization)")
    print("  - comparison_timeseries.csv (detailed time-series data)")
    print("  - comparison_summary.csv (summary statistics)")


if __name__ == "__main__":
    main()
