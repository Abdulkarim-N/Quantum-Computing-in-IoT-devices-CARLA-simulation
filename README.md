# Quantum Computing in Autonomous Vehicles - CARLA Demo

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.16-orange.svg)](https://carla.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-Latest-purple.svg)](https://qiskit.org/)

A demo project exploring quantum algorithms for autonomous vehicle decision-making. I simulated Grover's Algorithm running on classical hardware and compared it to traditional approaches in a realistic driving scenario.

**Author**: Abdulkarim-N  
**Demo**: Vehicle Cut-In Scenario

---

## What This Project Does

This is a proof-of-concept that tests whether quantum computing could improve how self-driving cars make split-second decisions. I built two versions of the same emergency scenario:

1. **Classical Version**: Standard if-then logic (what most cars use today)
2. **Quantum Version**: Grover's algorithm simulated on regular hardware

Both were tested in CARLA, an open-source driving simulator, under identical dangerous conditions.

---

## The Scenario

Imagine you're driving on the highway when another car suddenly cuts in front of you. You have milliseconds to react. Should you:
- Keep going?
- Brake gently?
- Slam the brakes?
- Swerve left?

My algorithms had to make this choice in real-time while avoiding a crash.

**Setup**:
- My car (ego vehicle) cruising at 18 m/s on the highway
- Another car (NPC) starts 60m behind in the right lane
- NPC speeds up to 24 m/s, passes me, then cuts directly in front
- Both algorithms had to react to avoid collision

---

## What I Found

| What I Measured | Classical | Quantum (Simulated) |
|----------------|-----------|---------------------|
| **Closest Distance** | 3.35m | 3.31m |
| **Response Time** | 0.10ms | 5.23ms |
| **Both Avoided Crash?** | ✅ Yes | ✅ Yes |
| **Real-time Ready?** | ✅ Yes | ❌ Not yet |

**Bottom Line**: The classical algorithm was faster and slightly safer for this scenario. The quantum approach worked but was 52x slower because I was simulating quantum hardware on a regular computer. On actual quantum hardware, this would be different.

---

## Watch It In Action

- [Classical Algorithm Demo](https://drive.google.com/file/d/1WTzEKVel96NhhfbY9kMS5CGP-LlEJ2oA/view?usp=sharing) - Traditional decision-making
- [Quantum Algorithm Demo](https://drive.google.com/file/d/1JKdsNiRFhch4yOJXmQUkKBstCZX_rJCV/view?usp=sharing) - Grover's algorithm in action

Both videos show the exact same cut-in maneuver with different decision-making methods.

---

## Tech Stack

- **CARLA Simulator 0.9.16** - Realistic autonomous driving environment
- **Qiskit + Aer Simulator** - Quantum computing framework (simulated on classical hardware)
- **Python 3.8+** - Main programming language
- **NumPy, Matplotlib, Pandas** - Data analysis and visualization

---

## How It Works

### Classical Algorithm
Straightforward decision tree:
- If car is too close → hard brake
- If car is cutting in and time is tight → hard brake
- If car is ahead and close → gentle brake
- Otherwise → maintain speed

**Pros**: Super fast (0.1ms), predictable, easy to verify  
**Cons**: Doesn't explore all options, limited by predefined rules

### Quantum Algorithm (Grover's Search)
Uses quantum mechanics to explore all options simultaneously:
1. Calculates a "cost" for each possible action
2. Creates quantum superposition of all actions
3. Uses Grover's algorithm to amplify the best choice
4. Measures the result (most likely = best action)

**Pros**: Explores entire solution space, theoretically faster for large problems  
**Cons**: Simulation overhead makes it slow on classical hardware, more complex

---

## Running the Demo

### Prerequisites
1. Download [CARLA 0.9.16](https://github.com/carla-simulator/carla/releases/tag/0.9.16)
2. Install Python 3.8+
3. Install dependencies: `pip install -r requirements.txt`

### Steps

```bash
# 1. Start CARLA server
cd /path/to/CARLA_0.9.16
./CarlaUE4.sh

# 2. Run classical version
python src/scenarios/scenario2_vehicle_cutin.py

# 3. Run quantum version (this will be slower)
python src/scenarios/scenario2_vehicle_cutin_quantum.py

# 4. Compare results
python compare_algorithms.py
```

The comparison script generates 6 plots, CSV data, and a detailed report in the `results/` folder.

---

## What I Learned

**Why Classical Won**: 
- The simulation overhead of quantum computing killed its speed advantage
- For simple scenarios with few options, classical is hard to beat
- Real-time systems need sub-10ms response times

**When Quantum Might Help**:
- Complex scenarios with many vehicles and options
- When running on actual quantum hardware (no simulation overhead)
- Large action spaces (16+ choices) where quantum scales better

**The Reality**: This was a simulation of quantum computing on regular hardware. The 5.2ms delay is mostly from simulating qubits with classical bits. On real quantum processors, this could be much faster.

---

## Future Work

- Test on actual quantum hardware when available
- Try more complex scenarios (multiple vehicles, intersections)
- Explore hybrid approaches (quantum planning + classical control)
- Scale up the action space to see where quantum advantages emerge

---

## Author

**Abdulkarim-N**  
GitHub: [@Abdulkarim-N](https://github.com/Abdulkarim-N)

This was a capstone project exploring the intersection of quantum computing and autonomous vehicles. Feel free to reach out with questions!

---

## License

MIT License - See LICENSE file for details

---

**Status**: ✅ Demo Completed (Phase 1)  
**Note**: This project simulates quantum algorithms on classical hardware. Results represent simulated quantum behavior, not actual quantum hardware performance.
