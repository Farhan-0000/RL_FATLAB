# VIT BCSE432E RL Lab Exam — One Template, All 12 Official Labs

Every program in this guide is tailored to your **exact VIT Lab Assignment PDFs (Labs 1–12)** while following the **same 4-block structure** and the **same variable names** (`state, action, next_state, reward, alpha, gamma, epsilon, Q, V, actor, critic`).

---

## 1. The Master Template (Memorize This Skeleton Once)

```python
# 1. SETUP                      -> alpha, gamma, epsilon, episodes
# 2. STATES AND ACTIONS         -> env / grid / zones, Q or V table (or actor/critic networks)
# 3. TRAINING LOOP
for episode in range(episodes):
    state = start               # (or: state, _ = env.reset())
    while not done:
        # POLICY                -> how to choose action (epsilon-greedy or actor network)
        # ENVIRONMENT           -> compute next_state, reward, done
        # EQUATION              -> the learning update  <-- ONLY THIS CHANGES PER LAB!
        state = next_state
# 4. OUTPUT                     -> print V, Q, policy, or comparison metrics
```

---

## 2. Master Cheat Sheet: What Changes Across the 12 Official Labs?

| Lab | Official Problem Statement | Data Structure | The Core Update Equation (`# EQUATION`) |
| :--- | :--- | :--- | :--- |
| **Lab 1** | **Ad Selection ($\epsilon$-Greedy Variants + Regret)** | `Q = np.zeros(n_ads)` | `Q[a] += (1 / N[a]) * (reward - Q[a])` |
| **Lab 2** | **$10 \times 10$ Stochastic Grid (Value Iteration)** | `V = np.zeros((10, 10))` | `V[s] = max_a sum_s' P(s'|s,a) * (r + gamma * V[s'])` |
| **Lab 3** | **7-Zone Taxi Fleet (Policy Iteration)** | `V = np.zeros(7)`, `policy` | **Eval:** `V[s] = E[r + gamma * V[s']]` $\rightarrow$ **Improve:** `argmax_a` |
| **Lab 4** | **Monte Carlo Prediction (`S1..S3` & $4 \times 4$ Grid)** | `V`, `returns_sum`, `counts` | Backward `G = r + gamma * G`, then First-Visit average of `G` |
| **Lab 5** | **$16 \times 16$ Delivery Drone (Q-Learning)** | `Q = defaultdict(zeros)` | `Q[s][a] += alpha * (r + gamma * np.max(Q[s']) - Q[s][a])` |
| **Lab 6** | **4-Way Traffic Light Control (SARSA)** | `Q = np.zeros((..., n_actions))` | `Q[s][a] += alpha * (r + gamma * Q[s'][next_a] - Q[s][a])` |
| **Lab 7** | **CartPole-v1 (Deep Q-Learning / DQN)** | `model = nn.Sequential(...)` | `loss = (model(x)[a] - (r + gamma * (1-done) * model(nx).max()))**2` |
| **Lab 8** | **TD-Gammon (1-Ply vs. 2-Ply Search)** | `critic = nn.Sequential(...)` | `td_error = r + gamma * V(s') - V(s)` + eligibility trace update |
| **Lab 9** | **CartPole-v1 (REINFORCE vs. DQN)** | `actor = nn.Sequential(...)` | `loss = -(torch.stack(log_probs) * returns).mean()` |
| **Lab 10** | **Hybrid Elevator Control (Actor-Critic)** | `actor` (discrete+cont), `critic` | `adv = r + gamma * V(s') - V(s)`; `loss = -log_prob * adv + adv**2` |
| **Lab 11** | **Autonomous Delivery Vehicle (Simple vs. A2C)** | `actor` + `critic` | **Simple:** 1-step TD `adv` vs. **Adv (A2C):** `returns - V(s)` + entropy |
| **Lab 12** | **Real-World App / Vehicle Control (PPO)** | `actor` + `critic` | `ratio = exp(new_lp - old_lp)`; `-min(ratio*adv, clip(ratio)*adv)` |

---

## LAB-1: Epsilon-Greedy Variants for Choosing Advertisements & Regret

### Theory & Equations Required by the Lab Sheet
* **Incremental Sample-Average Update:** Instead of a fixed step size $\alpha$, the exact sample mean of rewards for Advertisement $A$ is maintained using the pull count $N(A)$:
  $$Q(A)_{\text{new}} = Q(A)_{\text{old}} + \frac{1}{N(A)} \left[ R - Q(A)_{\text{old}} \right]$$
* **Four $\epsilon$-Decay Variants ($t = 0, 1, \dots, T-1$):**
  1. **Linear Decay:** $\epsilon_t = \epsilon_{\text{start}} - \frac{\epsilon_{\text{start}} - \epsilon_{\text{end}}}{T} \times t$
  2. **Exponential Decay:** $\epsilon_t = \epsilon_{\text{end}} + (\epsilon_{\text{start}} - \epsilon_{\text{end}}) \times e^{-\lambda t}$
  3. **Step Decay:** $\epsilon_t = \max\left(\epsilon_{\text{end}}, \, \epsilon_{\text{start}} \times \text{drop}^{\lfloor t / \text{step\_size} \rfloor}\right)$
  4. **Inverse Time Decay:** $\epsilon_t = \frac{\epsilon_{\text{start}}}{1 + \text{decay\_rate} \times t}$
* **Cumulative Regret over $T$ Trials:** Measures total expected click loss from not always picking the best ad $\mu^* = \max_a \mu_a$:
  $$R(T) = \sum_{t=1}^{T} \left(\mu^* - \mu_{a_t}\right)$$

### Code (All 4 Decay Variants + Regret in One Loop)
```python
import numpy as np
import random
import math

# 1. SETUP
eps_start, eps_end = 1.0, 0.01
episodes = 1000                                         # minimum 500 trials required
true_means = [0.15, 0.55, 0.35, 0.25]                   # true click rates (mu) of 4 Ads
best_mu = max(true_means)                               # mu* (optimal expected reward)
n_actions = len(true_means)

# Helper function for the 4 epsilon decay variants from the lab sheet
def get_epsilon(mode, t):
    if mode == "standard":    return 0.1
    if mode == "linear":      return eps_start - ((eps_start - eps_end) / episodes) * t
    if mode == "exponential": return eps_end + (eps_start - eps_end) * math.exp(-0.005 * t)
    if mode == "step":        return max(eps_end, eps_start * (0.5 ** (t // 200)))
    if mode == "inverse":     return eps_start / (1 + 0.01 * t)

# 2 & 3. RUN TRAINING LOOP FOR EACH VARIANT
for mode in ["standard", "linear", "exponential", "step", "inverse"]:
    Q = np.zeros(n_actions)                             # Q(A): estimated value of each Ad
    N = np.zeros(n_actions)                             # N(A): number of times Ad A was chosen
    total_regret = 0.0

    for t in range(episodes):
        epsilon = get_epsilon(mode, t)

        # POLICY: epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = np.argmax(Q)

        # ENVIRONMENT: simulate user click (1 or 0) and accumulate Regret
        reward = 1 if random.random() < true_means[action] else 0
        total_regret += (best_mu - true_means[action])  # R(T) += (mu* - mu_a)

        # EQUATION: Q(A)_new = Q(A)_old + (1 / N(A)) * (R - Q(A)_old)
        N[action] += 1
        Q[action] += (1.0 / N[action]) * (reward - Q[action])

    # 4. OUTPUT
    print(f"{mode.upper():11s} | Best Ad: {np.argmax(Q)} | Q: {np.round(Q, 2)} | Total Regret: {total_regret:.2f}")
```

---

## LAB-2: $10 \times 10$ Grid World — Value Iteration & Discount Factor Analysis

### Theory & Exam Observations
* **Stochastic MDP:** The grid has 100 states $(r, c) \in [0, 9] \times [0, 9]$ with start $(0,0)$ and terminal goal $(9,9)$. Taking action $a \in \{\text{Up, Down, Left, Right}\}$ moves in direction $a$ with probability $0.8$, and in each of the other $3$ directions with probability $\frac{0.2}{3}$.
* **Bellman Optimality Update (Value Iteration):**
  $$V_{k+1}(s) = \max_{a} \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V_k(s') \right]$$
* **Observations to Write in Your Exam for $\gamma = 0.9$, $\gamma = 0$, and $\gamma = 1$:**
  * **$\gamma = 0.9$ (Balanced):** Values increase smoothly toward $(9,9)$, producing a shortest-path policy to the goal.
  * **$\gamma = 0.0$ (Myopic / Short-Sighted):** The agent only cares about the immediate 1-step reward. States directly adjacent to $(9,9)$ have positive value (since they can reach $+10$ in one step), while all other states have $V(s) = -1.0$ and cannot see the distant goal.
  * **$\gamma = 1.0$ (Undiscounted):** Since $(9,9)$ is an absorbing terminal goal and non-terminal steps cost $-1$, Value Iteration still converges because the agent wants to minimize the number of $-1$ steps before hitting $+10$, though convergence takes more iterations than $\gamma = 0.9$.

### Code
```python
import numpy as np

# 1. SETUP
grid_size = 10
goal = (9, 9)
threshold = 1e-4
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]            # 0:Up, 1:Down, 2:Left, 3:Right
arrows  = ["U", "D", "L", "R"]

# Helper to step on the 10x10 grid (clamped to walls)
def get_next_state(r, c, action_idx):
    dr, dc = actions[action_idx]
    nr = max(0, min(grid_size - 1, r + dr))
    nc = max(0, min(grid_size - 1, c + dc))
    return (nr, nc)

# 2 & 3. VALUE ITERATION FUNCTION (run for gamma = 0.9, 0.0, 1.0)
def run_value_iteration(gamma):
    V = np.zeros((grid_size, grid_size))
    policy = np.zeros((grid_size, grid_size), dtype=int)

    while True:
        delta = 0
        new_V = np.copy(V)

        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) == goal:
                    continue                            # terminal state: value stays 0

                # Evaluate all 4 actions using 0.8 intended + 0.2/3 other transitions
                q_values = []
                for a in range(4):
                    expected_val = 0.0
                    for actual_a in range(4):
                        prob = 0.8 if actual_a == a else (0.2 / 3.0)
                        nr, nc = get_next_state(r, c, actual_a)
                        reward = 10.0 if (nr, nc) == goal else -1.0
                        expected_val += prob * (reward + gamma * V[nr, nc])
                    q_values.append(expected_val)

                # EQUATION: Bellman Optimality Backup
                new_V[r, c] = max(q_values)
                policy[r, c] = np.argmax(q_values)
                delta = max(delta, abs(new_V[r, c] - V[r, c]))

        V = new_V
        if delta < threshold:
            break
    return V, policy

# 4. OUTPUT
for g in [0.9, 0.0, 1.0]:
    V_opt, pi_opt = run_value_iteration(g)
    print(f"\n=== GAMMA = {g} | V(0,0) = {V_opt[0,0]:.2f}, V(8,9) = {V_opt[8,9]:.2f} ===")
    if g == 0.9:
        print("Optimal Value Function V* (rounded):\n", np.round(V_opt, 1))
        print("\nOptimal Policy Grid:")
        for r in range(grid_size):
            row_str = [arrows[pi_opt[r, c]] if (r, c) != goal else "G" for c in range(grid_size)]
            print(" ".join(row_str))
```

---

## LAB-3: Policy Iteration for 7-Zone City Taxi Fleet Management

### Theory & Exam Comparison (Policy Iteration vs. Value Iteration)
* **MDP Setup:**
  * **States:** 7 City Zones `0..6` corresponding to `A, B, C, D, E, F, G`, each with a passenger demand probability $p_{\text{demand}}(s)$.
  * **Actions:** Either **Stay (`0`)** in the current zone to pick up a passenger, or **Move (`-1` or `+1`)** to an adjacent zone.
  * **Rewards:**
    * If **Stay (`0`)**: With probability $p_{\text{demand}}(s)$, pick up passenger ($+20$); with probability $1 - p_{\text{demand}}(s)$, fail to find passenger ($-5$).
    * If **Move (`-1` or `+1`)**: Costs $-1$ fuel/time; with probability $0.7$, moves to target zone $s'$; with probability $0.3$, traffic keeps the taxi in zone $s$.
* **Intuitions — Policy Iteration vs. Value Iteration:**
  * **Policy Iteration** alternates between *Policy Evaluation* (solving $V^\pi(s)$ for a fixed policy) and *Policy Improvement* ($\pi'(s) = \arg\max_a Q^\pi(s,a)$). It converges in very few outer iterations (often 3–5) because policies stabilize long before exact values do.
  * **Value Iteration** combines evaluation and improvement into a single $\max_a$ backup every sweep. Each sweep is faster, but it takes more sweeps to converge.

### Code
```python
import numpy as np

# 1. SETUP
gamma = 0.9
zones = ["A", "B", "C", "D", "E", "F", "G"]
n_states = len(zones)
demand_prob = [0.8, 0.4, 0.3, 0.1, 0.3, 0.5, 0.7]       # Zone A & G high demand, Zone D low
actions = [-1, 0, 1]                                    # 0: Move Left, 1: Stay/Pickup, 2: Move Right
action_names = ["Move Left", "Stay & Pickup", "Move Right"]

# Helper: Expected Bellman value for (state, action_idx) given V
def calc_action_value(s, a_idx, V):
    move = actions[a_idx]
    if move == 0:                                       # Action = Stay & try to pick up passenger
        p = demand_prob[s]
        exp_reward = p * 20.0 + (1 - p) * (-5.0)
        return exp_reward + gamma * V[s]
    else:                                               # Action = Move to neighbor zone
        target_s = max(0, min(n_states - 1, s + move))
        # 0.7 moves to target_s, 0.3 stays in s due to traffic; fuel cost = -1
        return -1.0 + gamma * (0.7 * V[target_s] + 0.3 * V[s])

# 2. STATES AND ACTIONS
V = np.zeros(n_states)
policy = np.zeros(n_states, dtype=int)                  # start with all 'Move Left' (0)

# 3. TRAINING LOOP: POLICY ITERATION
while True:
    # 3a. POLICY EVALUATION (compute V under current policy)
    while True:
        delta = 0
        for s in range(n_states):
            old_v = V[s]
            V[s] = calc_action_value(s, policy[s], V)
            delta = max(delta, abs(old_v - V[s]))
        if delta < 1e-4:
            break

    # 3b. POLICY IMPROVEMENT (update policy greedily)
    policy_stable = True
    for s in range(n_states):
        old_action = policy[s]
        q_vals = [calc_action_value(s, a, V) for a in range(len(actions))]
        policy[s] = np.argmax(q_vals)
        if old_action != policy[s]:
            policy_stable = False

    if policy_stable:
        break

# 4. OUTPUT
print("Zone | Demand | V*(Zone) | Optimal Action")
print("-" * 45)
for s in range(n_states):
    print(f"  {zones[s]}  |  {demand_prob[s]:.1f}   |  {V[s]:6.2f}  | {action_names[policy[s]]}")
```

---

## LAB-4: First-Visit Monte Carlo Prediction (Part 1: 3-State MDP & Part 2: $4 \times 4$ Gridworld)

### Theory & Intuition
* **First-Visit Monte Carlo Prediction:** Evaluates a fixed policy $\pi$ solely from simulated episodes without needing transition probabilities $P(s' \mid s, a)$.
* **Algorithm:**
  1. Simulate a complete episode $(s_0, r_0), (s_1, r_1), \dots, (s_{T-1}, r_{T-1})$.
  2. Compute the discounted return $G_t = r_t + \gamma G_{t+1}$ for each step $t$.
  3. **First-Visit Rule:** Only record $G_t$ for state $s_t$ if $s_t$ did **not** appear at an earlier time step $0, \dots, t-1$ in the same episode.
  4. Average the recorded returns across all episodes: $V(s) = \frac{\text{returns\_sum}[s]}{\text{counts}[s]}$.

### Code (Both Part 1 and Part 2 from the Lab Sheet)
```python
import numpy as np
import random

# =====================================================================
# PART 1: 3-State MDP (S1, S2, S3) under Fixed Policy "Always Right"
# =====================================================================
gamma_1 = 0.9
episodes_1 = 1000
returns_sum_1 = {"S1": 0.0, "S2": 0.0, "S3": 0.0}
counts_1 = {"S1": 0, "S2": 0, "S3": 0}

for ep in range(episodes_1):
    # Generate episode starting from S1 or S2
    state = "S1"
    history = []
    for step in range(10):
        action = "Right"                                # Fixed policy: always Right
        if state == "S1":
            next_state, reward = ("S2", 0) if action == "Right" else ("S1", -1)
        elif state == "S2":
            next_state, reward = ("S3", 1) if action == "Right" else ("S1", 0)
        else:                                           # S3 absorbs with 0 reward
            next_state, reward = ("S3", 0)
        history.append((state, reward))
        state = next_state
        if state == "S3" and step >= 2:
            break

    # First-Visit Monte Carlo Return Calculation
    G = 0.0
    G_at_step = [0.0] * len(history)
    for t in reversed(range(len(history))):
        G = history[t][1] + gamma_1 * G
        G_at_step[t] = G

    visited = set()
    for t, (s, _) in enumerate(history):
        if s not in visited:                            # First-visit check
            visited.add(s)
            returns_sum_1[s] += G_at_step[t]
            counts_1[s] += 1

print("=== PART 1: 3-State MDP Estimated V(s) ===")
for s in ["S1", "S2", "S3"]:
    v_est = returns_sum_1[s] / max(1, counts_1[s])
    print(f"V({s}) = {v_est:.4f}")

# =====================================================================
# PART 2: 4x4 Stochastic Gridworld (S0..S15) under Random Policy
# =====================================================================
gamma_2 = 0.95
episodes_2 = 10000
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]              # Up, Down, Left, Right
returns_sum_2 = np.zeros(16)
counts_2 = np.zeros(16)

for ep in range(episodes_2):
    state = 0                                           # Start at S0 (or random start for full coverage)
    history = []

    while state != 15 and len(history) < 200:           # S15 is terminal
        # POLICY: uniformly random action
        intended_a = random.randint(0, 3)

        # ENVIRONMENT: 0.8 intended direction, 0.2 random other direction
        if random.random() < 0.8:
            actual_a = intended_a
        else:
            others = [a for a in range(4) if a != intended_a]
            actual_a = random.choice(others)

        r, c = divmod(state, 4)
        nr, nc = r + moves[actual_a][0], c + moves[actual_a][1]

        if nr < 0 or nr >= 4 or nc < 0 or nc >= 4:      # Off-grid: stays in same state, reward -1
            next_state = state
            reward = -1.0
        else:
            next_state = nr * 4 + nc
            reward = 10.0 if next_state == 15 else 0.0

        history.append((state, reward))
        state = next_state

    # EQUATION: First-Visit Monte Carlo Return Calculation
    G = 0.0
    G_at_step = [0.0] * len(history)
    for t in reversed(range(len(history))):
        G = history[t][1] + gamma_2 * G
        G_at_step[t] = G

    visited = set()
    for t, (s, _) in enumerate(history):
        if s not in visited:
            visited.add(s)
            returns_sum_2[s] += G_at_step[t]
            counts_2[s] += 1

V_grid = np.divide(returns_sum_2, counts_2, out=np.zeros(16), where=counts_2 > 0)
print("\n=== PART 2: 4x4 Gridworld V(s) Table ===")
print(np.round(V_grid.reshape(4, 4), 2))
```

---

## LAB-5: $16 \times 16$ Delivery Drone Environment with Tabular Q-Learning

### Theory & How to Memorize This Quickly
* Even though the environment has several rules (battery, recharge stations, wind region, obstacles, and 2 sequential packages), the **Training Loop and Q-Learning Equation are 100% identical to basic Q-Learning**.
* **State Representation:** `(r, c, b, pkg_stage)` where:
  * `(r, c)` is the drone position on the $16 \times 16$ grid.
  * `b` is battery level (up to `40`).
  * `pkg_stage` tracks sequential package progress (`0`: going to P1 pickup `(0,15)`, `1`: carrying P1 to drop-off `(15,0)`, `2`: going to P2 pickup `(15,15)`, `3`: carrying P2 to final drop-off `(7,12)`).
* **Exponential Decay Formula:** $\epsilon_t = \max(0.01, \, 1.0 \cdot e^{-0.00008 t})$

### Code
```python
import numpy as np
import random
import math
from collections import defaultdict

# 1. SETUP
alpha = 0.1
gamma = 0.99
eps_0, eps_min, lam = 1.0, 0.01, 0.00008
episodes = 10000

# 2. STATES AND ACTIONS (16x16 Delivery Drone Environment)
moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]              # 0:Up, 1:Down, 2:Left, 3:Right
obstacles = {(3, 8), (4, 8), (5, 8), (10, 2), (10, 3), (10, 4), (10, 5)}
rechargers = {(4, 4), (11, 11)}
targets = [(0, 15), (15, 0), (15, 15), (7, 12)]         # P1_pick, P1_drop, P2_pick, P2_drop
Q = defaultdict(lambda: np.zeros(4))                    # Tabular dictionary Q[(state)][action]

def drone_step(state, action):
    r, c, b, stage = state
    # Wind region: r, c in [6, 9] -> 25% chance of drifting 1 cell Down instead
    dr, dc = (1, 0) if (6 <= r <= 9 and 6 <= c <= 9 and random.random() < 0.25) else moves[action]
    nr, nc = r + dr, c + dc
    b -= 1
    reward = -1.0

    # Wall or Obstacle Collision: -10 penalty, stays in current position
    if nr < 0 or nr >= 16 or nc < 0 or nc >= 16 or (nr, nc) in obstacles:
        nr, nc = r, c
        reward -= 10.0

    # Recharging Station: restores b = 40 and gives +5 if b < 40
    if (nr, nc) in rechargers:
        if b < 40:
            reward += 5.0
        b = 40

    # Sequential Package Pickup / Drop-off logic
    if (nr, nc) == targets[stage]:
        if stage == 1: reward += 100.0                  # Package 1 delivered
        if stage == 3: reward += 300.0                  # Package 2 delivered (Terminal Goal)
        stage += 1

    done = (stage == 4) or (b <= 0)
    if b <= 0 and stage < 4:
        reward -= 200.0                                 # Out of battery penalty

    return (nr, nc, b, stage), reward, done

# 3. TRAINING & EVALUATION LOOP
total_steps = 0
for episode in range(1, episodes + 1):
    state = (0, 0, 40, 0)                               # Start (0,0), battery 40, stage 0
    done = False

    while not done:
        epsilon = max(eps_min, eps_0 * math.exp(-lam * total_steps))
        total_steps += 1

        # POLICY: epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = int(np.argmax(Q[state]))

        # ENVIRONMENT
        next_state, reward, done = drone_step(state, action)

        # EQUATION: Q-Learning Update
        best_next = 0.0 if done else np.max(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
        state = next_state

    # 4. OUTPUT: Evaluate every 500 episodes (100-episode greedy rollout with epsilon = 0)
    if episode % 2000 == 0:                             # (use 500 in exam if desired)
        eval_rewards, completions = [], 0
        for _ in range(100):
            s, d, ep_r = (0, 0, 40, 0), False, 0.0
            while not d:
                a = int(np.argmax(Q[s]))
                s, r, d = drone_step(s, a)
                ep_r += r
            eval_rewards.append(ep_r)
            if s[3] == 4: completions += 1
        print(f"Ep {episode:5d} | Mean Eval Reward: {np.mean(eval_rewards):7.2f} | Completion Rate: {completions}%")
```

---

## LAB-6: 4-Way Traffic Light Control using SARSA (with Poisson Arrivals)

### Theory & What Changes from Lab 5
* **Problem:** Control a 4-way intersection (`NS` lanes vs. `EW` lanes) where cars arrive stochastically via a **Poisson distribution** (`np.random.poisson`).
* **State:** `(ns_queue_bin, ew_queue_bin, current_green_phase)`
* **Action:** `0` = Keep current green phase, `1` = Switch green phase to the other direction.
* **SARSA Update (On-Policy):** Picks `next_action` *before* updating and uses `Q[next_state][next_action]`.
* **Evaluation:** Compares average waiting queues of SARSA at `5000` and `10000` episodes against a **Fixed-Cycle Traffic Light Baseline** (which blindly switches every 5 steps).

### Code
```python
import numpy as np
import random

# 1. SETUP
alpha = 0.1
gamma = 0.95
epsilon = 0.1

# Environment step helper with Poisson vehicle arrivals
def traffic_step(ns_q, ew_q, green, action):
    if action == 1:                                     # Action 1 = switch light, 0 = keep light
        green = 1 - green

    # Poisson arrivals (mean 1.2 cars/step NS, 1.0 cars/step EW)
    ns_q = min(10, ns_q + np.random.poisson(1.2))
    ew_q = min(10, ew_q + np.random.poisson(1.0))

    # Discharging cars on the green phase (up to 3 cars pass per step)
    if green == 0:
        ns_q = max(0, ns_q - 3)
    else:
        ew_q = max(0, ew_q - 3)

    waiting_cars = ns_q + ew_q
    reward = -float(waiting_cars)                       # Minimize total queue / waiting time
    next_state = (ns_q, ew_q, green)
    return next_state, reward, waiting_cars

# 2 & 3. TRAIN SARSA FOR DIFFERENT EPISODE COUNTS (5000 vs 10000)
def train_sarsa(episodes):
    Q = np.zeros((11, 11, 2, 2))                        # (ns_q, ew_q, green, action)

    for episode in range(episodes):
        state = (2, 2, 0)                               # initial queues and green phase

        # POLICY: first action (SARSA 1)
        action = random.randint(0, 1) if random.random() < epsilon else int(np.argmax(Q[state]))

        for step in range(30):                          # 30 time-steps per simulated day
            # ENVIRONMENT
            next_state, reward, _ = traffic_step(state[0], state[1], state[2], action)

            # POLICY: next action (SARSA 2)
            next_action = random.randint(0, 1) if random.random() < epsilon else int(np.argmax(Q[next_state]))

            # EQUATION: SARSA Update (SARSA 3)
            Q[state][action] += alpha * (
                reward + gamma * Q[next_state][next_action] - Q[state][action]
            )
            state, action = next_state, next_action
    return Q

# Evaluate a learned Q-table vs Fixed-Cycle Baseline
def evaluate_policy(Q=None):
    total_wait = 0
    for _ in range(200):
        state = (2, 2, 0)
        for step in range(30):
            if Q is None:                               # Fixed-Cycle Baseline: switch every 5 steps
                action = 1 if (step % 5 == 0) else 0
            else:
                action = int(np.argmax(Q[state]))
            state, _, wait = traffic_step(state[0], state[1], state[2], action)
            total_wait += wait
    return total_wait / (200 * 30)

# 4. OUTPUT
print(f"Fixed-Cycle Baseline Avg Waiting Queue: {evaluate_policy(None):.2f} cars")
for eps in [5000, 10000]:
    learned_Q = train_sarsa(eps)
    print(f"SARSA ({eps:5d} episodes) Avg Waiting Queue: {evaluate_policy(learned_Q):.2f} cars")
```

---

## LAB-7: Deep Q-Learning (DQN) for CartPole

### Theory & Intuition
* **Why DQN?** `CartPole-v1` has a continuous 4-dimensional state space (`cart_pos, cart_vel, pole_angle, pole_vel`), so we replace the tabular Q-table with a PyTorch Neural Network `model(x)` that outputs 2 Q-values (`Left`, `Right`).
* **Experience Replay Buffer (`deque`):** Stores transitions `(x, action, reward, next_x, done)` and trains on random mini-batches to break temporal correlations between consecutive steps.
* **DQN Loss:**
  $$\mathcal{L}(\theta) = \left( Q_\theta(s, a) - \left[ r + \gamma (1 - \text{done}) \max_{a'} Q_\theta(s', a') \right] \right)^2$$

### Code
```python
import gymnasium as gym
import torch
import torch.nn as nn
import random
from collections import deque

# 1. SETUP
gamma = 0.99
epsilon = 0.15
episodes = 150
memory = deque(maxlen=2000)

# 2. STATES AND ACTIONS
env = gym.make("CartPole-v1")                           # 4 continuous state inputs, 2 actions
n_actions = 2
model = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, n_actions)
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 3. TRAINING LOOP
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        x = torch.tensor(state, dtype=torch.float32)

        # POLICY: epsilon-greedy on Q-network
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = model(x).argmax().item()

        # ENVIRONMENT
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_x = torch.tensor(next_state, dtype=torch.float32)
        total_reward += reward

        # EQUATION: DQN with Experience Replay
        memory.append((x, action, reward, next_x, float(done)))

        if len(memory) >= 32:
            batch = random.sample(memory, 32)
            loss = 0.0
            for bx, ba, br, bnx, bd in batch:
                q_val = model(bx)[ba]
                target = br + gamma * (1.0 - bd) * model(bnx).max().detach()
                loss += (q_val - target) ** 2

            loss = loss / 32.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state

    if (episode + 1) % 30 == 0:
        print(f"Episode {episode + 1:3d} | DQN CartPole Reward: {total_reward}")

# 4. OUTPUT
env.close()
```

---

## LAB-8: TD-Gammon Learning (Hyperparameters & 1-Ply vs. 2-Ply Search)

### Theory & Exam Observations
* **Architecture (`net.py` + `train.py` Concept):**
  * **`net.py` Role:** A neural network (`critic`) that takes a board state $s$ and outputs $V_\theta(s) \in [0, 1]$, representing the probability that Player 1 (`X`) wins from that board position.
  * **`train.py` Role:** Self-play training loop using Temporal Difference learning TD($\lambda$) with eligibility traces:
    $$\delta_t = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t), \quad e_t = \gamma \lambda e_{t-1} + \nabla_\theta V_\theta(s_t)$$
* **1-Ply vs. 2-Ply Move Selection (Exam Question):**
  * **1-Ply Search:** The agent looks **1 move ahead** (its own move) and picks the next board state $s'$ that maximizes $V_\theta(s')$:
    $$a_{\text{1-ply}} = \arg\max_{a} V_\theta(\text{afterstate}(s, a))$$
  * **2-Ply Search:** The agent looks **2 moves ahead** (its own move + the opponent's best counter-move averaged over dice rolls) using Minimax:
    $$a_{\text{2-ply}} = \arg\max_{a} \, \mathbb{E}_{\text{dice}} \left[ \min_{a_{\text{opp}}} V_\theta(\text{afterstate}(s', a_{\text{opp}})) \right]$$
  * **Observation:** 2-ply search significantly outperforms 1-ply search because it avoids moves that look locally good after 1 turn but leave the checker exposed to a devastating opponent counter-move on the next turn.

### Code (Self-Play Training + 1-Ply vs. 2-Ply Evaluation)
```python
import torch
import torch.nn as nn
import random

# 1. SETUP & HYPERPARAMETERS
goal = 12                                               # simplified Backgammon race track (0..12)

# Helper: Generate legal next board states (p1_pos, p2_pos) given dice roll
def get_legal_next_states(state, dice, player):
    p1, p2 = state
    candidates = []
    for step in [dice, 1]:                              # choice: full dice move or cautious 1-step
        if player == 1:
            np1 = min(goal, p1 + step)
            # Hitting opponent blot sends opponent back to 0!
            np2 = 0 if (np1 == p2 and np1 < goal) else p2
            candidates.append((np1, np2))
        else:
            np2 = min(goal, p2 + step)
            np1 = 0 if (np2 == p1 and np2 < goal) else p1
            candidates.append((np1, np2))
    return candidates

# Move Selection: 1-Ply vs 2-Ply Search
def select_move(critic, state, dice, ply=1):
    candidates = get_legal_next_states(state, dice, player=1)
    if ply == 1:
        # 1-PLY: Maximize V(s') after agent's own move
        return max(candidates, key=lambda s: critic(torch.tensor([s[0]/goal, s[1]/goal], dtype=torch.float32)).item())
    else:
        # 2-PLY: Agent move + Opponent's best counter-move (Minimax over average dice=2)
        best_s, best_val = candidates[0], -1e9
        for s_after_agent in candidates:
            if s_after_agent[0] >= goal:
                return s_after_agent
            opp_Replies = get_legal_next_states(s_after_agent, dice=2, player=2)
            # Opponent picks reply that MINIMIZES Player 1's win probability V(s'')
            worst_reply_val = min(
                critic(torch.tensor([r[0]/goal, r[1]/goal], dtype=torch.float32)).item()
                for r in opp_Replies
            )
            if worst_reply_val > best_val:
                best_val, best_s = worst_reply_val, s_after_agent
        return best_s

# 2 & 3. TRAIN TD-GAMMON NETWORK WITH DIFFERENT HYPERPARAMETERS (alpha, lam)
def train_td_gammon(alpha=0.02, gamma=0.95, lam=0.7, episodes=300):
    critic = nn.Sequential(                             # net.py: Evaluates board position V(s)
        nn.Linear(2, 16), nn.ReLU(),
        nn.Linear(16, 1), nn.Sigmoid()
    )
    for ep in range(episodes):                          # train.py: TD(lambda) self-play loop
        state = (0, 0)
        traces = [torch.zeros_like(p) for p in critic.parameters()]
        while state[0] < goal and state[1] < goal:
            x = torch.tensor([state[0]/goal, state[1]/goal], dtype=torch.float32)

            # Agent (Player 1) move (1-ply during training)
            next_state = select_move(critic, state, random.randint(1, 3), ply=1)
            # Opponent (Player 2) move if game not over
            if next_state[0] < goal:
                opp_moves = get_legal_next_states(next_state, random.randint(1, 3), player=2)
                next_state = min(opp_moves, key=lambda s: critic(torch.tensor([s[0]/goal, s[1]/goal], dtype=torch.float32)).item())

            done = (next_state[0] >= goal) or (next_state[1] >= goal)
            reward = 1.0 if next_state[0] >= goal else 0.0
            next_x = torch.tensor([next_state[0]/goal, next_state[1]/goal], dtype=torch.float32)

            # EQUATION: TD(lambda) Eligibility Trace Update
            value = critic(x)
            next_val = 0.0 if done else critic(next_x).detach()
            td_error = (reward + gamma * next_val - value).detach()

            critic.zero_grad()
            value.backward()
            with torch.no_grad():
                for param, trace in zip(critic.parameters(), traces):
                    trace.mul_(gamma * lam).add_(param.grad)
                    param.add_(alpha * td_error * trace)
            state = next_state
    return critic

# 4. OUTPUT: Compare Hyperparameters and 1-Ply vs 2-Ply Win Rate
for alpha, lam in [(0.01, 0.5), (0.02, 0.8)]:
    net = train_td_gammon(alpha=alpha, lam=lam)
    for ply in [1, 2]:
        wins = 0
        for _ in range(100):
            s = (0, 0)
            while s[0] < goal and s[1] < goal:
                s = select_move(net, s, random.randint(1, 3), ply=ply)
                if s[0] < goal:
                    s = random.choice(get_legal_next_states(s, random.randint(1, 3), player=2))
            if s[0] >= goal: wins += 1
        print(f"Hyperparams (alpha={alpha}, lambda={lam}) | {ply}-Ply Search Win Rate: {wins}%")
```

---

## LAB-9: CartPole using REINFORCE (Evaluated Against a DQN Baseline)

### Theory & Exam Comparison
* **REINFORCE (Monte Carlo Policy Gradient):** Directly optimizes a parameterized stochastic policy network $\pi_\theta(a \mid s)$ (`actor`) using full-episode discounted returns $G_t$:
  $$\mathcal{L}_{\text{REINFORCE}}(\theta) = -\frac{1}{T} \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) \cdot \hat{G}_t$$
* **REINFORCE vs. DQN Baseline Observations:**
  * **DQN (Value-Based, Off-Policy):** Learns action-values $Q_\theta(s, a)$ at every step using experience replay; sample-efficient for discrete action spaces, though sensitive to $\epsilon$ and target instability.
  * **REINFORCE (Policy-Based, On-Policy):** Directly learns action probabilities without a value table and naturally handles stochastic policies, but exhibits higher variance because it waits until the end of the episode to compute Monte Carlo returns $G_t$.

### Code (Trains Both REINFORCE and DQN Baseline on CartPole-v1)
```python
import gymnasium as gym
import torch
import torch.nn as nn
import random
from collections import deque

# 1. SETUP
gamma = 0.99
episodes = 150
env = gym.make("CartPole-v1")
n_actions = 2

# =====================================================================
# MODEL A: REINFORCE (Policy Gradient)
# =====================================================================
actor = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, n_actions), nn.Softmax(dim=-1)
)
opt_reinforce = torch.optim.Adam(actor.parameters(), lr=0.005)
reinforce_rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    log_probs, rewards = [], []
    done = False

    while not done:
        x = torch.tensor(state, dtype=torch.float32)

        # POLICY: sample from actor network
        dist = torch.distributions.Categorical(actor(x))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        # ENVIRONMENT
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    # EQUATION: REINFORCE
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = -(torch.stack(log_probs) * returns).mean()
    opt_reinforce.zero_grad()
    loss.backward()
    opt_reinforce.step()
    reinforce_rewards.append(sum(rewards))

# =====================================================================
# MODEL B: Deep Q-Learning (DQN) Baseline
# =====================================================================
q_net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, n_actions))
opt_dqn = torch.optim.Adam(q_net.parameters(), lr=0.005)
memory = deque(maxlen=2000)
dqn_rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    ep_reward, done = 0, False
    while not done:
        x = torch.tensor(state, dtype=torch.float32)
        action = random.randint(0, 1) if random.random() < 0.15 else q_net(x).argmax().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_x = torch.tensor(next_state, dtype=torch.float32)
        memory.append((x, action, reward, next_x, float(done)))
        ep_reward += reward

        if len(memory) >= 32:
            batch = random.sample(memory, 32)
            loss = sum((q_net(bx)[ba] - (br + gamma * (1 - bd) * q_net(bnx).max().detach())) ** 2
                       for bx, ba, br, bnx, bd in batch) / 32.0
            opt_dqn.zero_grad(); loss.backward(); opt_dqn.step()
        state = next_state
    dqn_rewards.append(ep_reward)

# 4. OUTPUT: Compare REINFORCE vs DQN Baseline
print(f"REINFORCE Avg Reward (Last 30 eps): {sum(reinforce_rewards[-30:]) / 30:.2f}")
print(f"DQN Baseline Avg Reward (Last 30):  {sum(dqn_rewards[-30:]) / 30:.2f}")
env.close()
```

---

## LAB-10: Hybrid Elevator Control Agent using Actor-Critic (Discrete + Continuous)

### Theory & Intuition
* **Why Hybrid Actor-Critic?** Your Lab 10 sheet specifies that the elevator agent must simultaneously make:
  1. **Discrete action decisions:** Direction $a_{\text{disc}} \in \{\text{Down } (-1), \text{Stay } (0), \text{Up } (+1)\}$ modeled via a `Categorical` distribution.
  2. **Continuous control decisions:** Elevator speed / timing $a_{\text{cont}} \in [0.5, 2.0]$ modeled via a Gaussian `Normal` distribution $\mathcal{N}(\mu_\theta(s), \sigma)$.
* **Joint Log-Probability:** Since both actions are chosen from the shared state representation, the total Actor log-probability is simply their sum:
  $$\log \pi_\theta(a_{\text{disc}}, a_{\text{cont}} \mid s) = \log \pi_\theta(a_{\text{disc}} \mid s) + \log \pi_\theta(a_{\text{cont}} \mid s)$$
* **Actor-Critic Update:** Uses the Critic's 1-step TD advantage $A_t = r + \gamma V_\phi(s') - V_\phi(s)$ to update both heads simultaneously.

### Code
```python
import torch
import torch.nn as nn
import random

# 1. SETUP
gamma = 0.95
episodes = 200
moves = [-1, 0, 1]                                      # Discrete: 0=Down, 1=Stay, 2=Up

# 2. STATES AND ACTIONS (Hybrid Actor + Critic Networks)
# State: [elevator_floor / 10, passenger_request_floor / 10]
shared_actor = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
discrete_head = nn.Sequential(nn.Linear(32, 3), nn.Softmax(dim=-1))     # Direction probabilities
continuous_mu = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())           # Speed factor in (0, 1) -> [0.5, 2.0]
critic = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))   # State value V(s)

params = (list(shared_actor.parameters()) + list(discrete_head.parameters()) +
          list(continuous_mu.parameters()) + list(critic.parameters()))
optimizer = torch.optim.Adam(params, lr=0.005)

# 3. TRAINING LOOP
for episode in range(episodes):
    elev_floor = 0.0
    req_floor = float(random.randint(1, 10))            # Passenger waiting at random floor 1..10
    total_wait_time = 0.0

    for step in range(25):
        x = torch.tensor([elev_floor / 10.0, req_floor / 10.0], dtype=torch.float32)

        # POLICY: Hybrid Actor (Discrete direction + Continuous speed)
        h = shared_actor(x)
        dist_disc = torch.distributions.Categorical(discrete_head(h))
        action_disc = dist_disc.sample()

        speed_mean = 0.5 + 1.5 * continuous_mu(h).squeeze()             # Speed in [0.5, 2.0]
        dist_cont = torch.distributions.Normal(speed_mean, 0.2)
        action_cont = torch.clamp(dist_cont.sample(), 0.5, 2.0)

        joint_log_prob = dist_disc.log_prob(action_disc) + dist_cont.log_prob(action_cont)

        # ENVIRONMENT: Elevator moves by (direction * speed); minimize passenger waiting time
        direction = moves[action_disc.item()]
        next_floor = max(0.0, min(10.0, elev_floor + direction * action_cont.item()))
        distance = abs(req_floor - next_floor)
        step_wait = distance / action_cont.item()
        total_wait_time += step_wait

        # Reward: negative waiting distance + energy cost of high speed (+10 bonus on arrival)
        done = distance < 0.8
        reward = (10.0 if done else -distance) - 0.2 * action_cont.item()
        next_x = torch.tensor([next_floor / 10.0, req_floor / 10.0], dtype=torch.float32)

        # EQUATION: Actor-Critic (1-Step TD Advantage)
        value = critic(x).squeeze()
        next_value = 0.0 if done else critic(next_x).squeeze().detach()
        advantage = (reward + gamma * next_value) - value

        loss = -joint_log_prob * advantage.detach() + advantage ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elev_floor = next_floor
        if done:
            break

    if (episode + 1) % 40 == 0:
        print(f"Episode {episode + 1:3d} | Final Floor: {elev_floor:4.1f}/{req_floor:2.0f} | Total Wait: {total_wait_time:.2f}")
```

---

## LAB-11: Autonomous Delivery Vehicle Fuel Optimization (Simple vs. Advanced Actor-Critic)

### Theory & Exam Comparison (Simple Actor-Critic vs. Advanced Actor-Critic)
* **Environment:** An autonomous delivery vehicle adjusts its speed (`0: Brake (-5)`, `1: Maintain (0)`, `2: Accelerate (+5)`) to match an eco-driving target speed dictated by traffic conditions and road slope while minimizing fuel consumption.
* **Simple Actor-Critic vs. Advanced Actor-Critic (A2C):**
  1. **Simple Actor-Critic (Online 1-Step TD):** Updates at every single time step using the 1-step bootstrapped TD error $A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. It is fast and online, but has higher bias early in training when the Critic $V_\phi$ is inaccurate.
  2. **Advanced Actor-Critic (A2C with Multi-Step Returns & Entropy Bonus):** Collects an episode rollout, computes normalized multi-step returns $G_t$, uses the baseline advantage $A_t = G_t - V(s_t)$, and adds an **Entropy Regularization bonus** $-\beta H(\pi_\theta)$ so the vehicle explores diverse driving actions before converging.

### Code (Both Simple and Advanced Actor-Critic on the Delivery Vehicle Task)
```python
import torch
import torch.nn as nn
import random

# 1. SETUP
gamma = 0.95
episodes = 200
moves = [-5, 0, 5]                                      # 0: Brake, 1: Maintain, 2: Accelerate
n_actions = len(moves)

# Delivery Vehicle Environment Step (speed, traffic_target_speed, road_slope)
def vehicle_step(speed, target_speed, slope, action):
    next_speed = max(0.0, min(100.0, speed + moves[action] - slope))
    speed_error = abs(target_speed - next_speed) / 10.0
    fuel_used = 0.5 * abs(moves[action]) + 0.1 * (next_speed / 20.0) ** 2
    reward = -(speed_error + 0.3 * fuel_used)           # Optimize fuel efficiency & target tracking
    return next_speed, reward, fuel_used

# 2 & 3. TRAIN SIMPLE (1-STEP TD) OR ADVANCED (EPISODE A2C + ENTROPY) ACTOR-CRITIC
def train_vehicle_ac(mode="simple"):
    actor = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, n_actions), nn.Softmax(dim=-1))
    critic = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.005)
    episode_rewards = []

    for episode in range(episodes):
        speed = 20.0
        target_speed = float(random.choice([40, 50, 60]))   # varying traffic condition
        slope = float(random.choice([-2, 0, 2]))            # varying road layout
        xs, log_probs, entropies, rewards = [], [], [], []

        for step in range(30):
            x = torch.tensor([speed / 100.0, target_speed / 100.0, slope / 5.0], dtype=torch.float32)

            # POLICY: actor
            dist = torch.distributions.Categorical(actor(x))
            action = dist.sample()

            # ENVIRONMENT
            next_speed, reward, _ = vehicle_step(speed, target_speed, slope, action.item())
            next_x = torch.tensor([next_speed / 100.0, target_speed / 100.0, slope / 5.0], dtype=torch.float32)
            rewards.append(reward)

            # EQUATION A: Simple Actor-Critic (updates every step using 1-step TD error)
            if mode == "simple":
                value = critic(x).squeeze()
                next_value = critic(next_x).squeeze().detach()
                advantage = (reward + gamma * next_value) - value
                loss = -dist.log_prob(action) * advantage.detach() + advantage ** 2
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            else:
                xs.append(x)
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())

            speed = next_speed

        # EQUATION B: Advanced Actor-Critic (updates after rollout with G_t - V(s) + Entropy)
        if mode == "advanced":
            G, returns = 0.0, []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)
            xs_t = torch.stack(xs)
            lp_t = torch.stack(log_probs)
            ent_t = torch.stack(entropies)
            ret_t = torch.tensor(returns, dtype=torch.float32)

            values = critic(xs_t).squeeze(-1)
            advantage = ret_t - values
            loss = -(lp_t * advantage.detach()).mean() + (advantage ** 2).mean() - 0.01 * ent_t.mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        episode_rewards.append(sum(rewards))
    return sum(episode_rewards[-30:]) / 30.0

# 4. OUTPUT
print(f"Simple Actor-Critic Avg Reward (Last 30 eps):   {train_vehicle_ac('simple'):.2f}")
print(f"Advanced Actor-Critic Avg Reward (Last 30 eps): {train_vehicle_ac('advanced'):.2f}")
```

---

## LAB-12: Proximal Policy Optimization (PPO) for a Real-World Task

### Theory & Intuition
* **Why PPO?** Standard Policy Gradient / Actor-Critic methods can become unstable if a single gradient update changes the policy probabilities too drastically.
* **How PPO Works (Only 3 Additions to Advanced Actor-Critic):**
  1. **Record Old Log-Probabilities (`# PPO 1`):** Store `old_log_probs = dist.log_prob(action).detach()` while collecting the episode.
  2. **Multiple Epochs per Rollout (`# PPO 2`):** Reuse the collected trajectory for `K = 4` optimization epochs, computing the probability ratio:
     $$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} = \exp\left(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t)\right)$$
  3. **Clipped Surrogate Objective (`# PPO 3`):** Clip $r_t(\theta)$ to $[1 - \epsilon, 1 + \epsilon] = [0.8, 1.2]$ and take the pessimistic `min` bound:
     $$\mathcal{L}^{\text{CLIP}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \, \text{clip}(r_t(\theta), 0.8, 1.2) A_t \right) \right]$$
* **Note on Real-World Task:** Below, PPO is applied to a **Smart Data-Center / Autonomous Cooling & Speed Controller** (and works identically on `CartPole-v1` if you swap the 4 environment lines!).

### Code
```python
import torch
import torch.nn as nn

# 1. SETUP
gamma = 0.95
episodes = 200

# 2. STATES AND ACTIONS (Real-World Smart Autonomous Speed/Cooling Controller)
target_level = 60.0                                     # Optimal target operating point
moves = [-5.0, 0.0, 5.0]                                # 0: Decrease, 1: Maintain, 2: Increase
n_actions = len(moves)

actor = nn.Sequential(
    nn.Linear(1, 32), nn.ReLU(),
    nn.Linear(32, n_actions), nn.Softmax(dim=-1)
)
critic = nn.Sequential(
    nn.Linear(1, 32), nn.ReLU(),
    nn.Linear(32, 1)
)
optimizer = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=0.003
)

# 3. TRAINING LOOP
for episode in range(episodes):
    state = 10.0                                        # Initial operating state
    xs, actions, old_log_probs, rewards = [], [], [], []

    for step in range(30):
        x = torch.tensor([state / 100.0], dtype=torch.float32)

        # POLICY: actor
        dist = torch.distributions.Categorical(actor(x))
        action = dist.sample()

        xs.append(x)
        actions.append(action)
        old_log_probs.append(dist.log_prob(action).detach())    # PPO 1: save old_log_prob

        # ENVIRONMENT
        next_state = max(0.0, min(100.0, state + moves[action.item()]))
        reward = -abs(target_level - next_state) / 10.0
        rewards.append(reward)
        state = next_state

    # EQUATION: PPO Clipped Surrogate Objective
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    xs = torch.stack(xs)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    for _ in range(4):                                          # PPO 2: 4 update epochs
        dist = torch.distributions.Categorical(actor(xs))
        new_log_probs = dist.log_prob(actions)
        values = critic(xs).squeeze(-1)
        advantage = returns - values

        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped = torch.clamp(ratio, 0.8, 1.2)                  # PPO 3: clip ratio to [0.8, 1.2]

        actor_loss = -torch.min(
            ratio * advantage.detach(),
            clipped * advantage.detach()
        ).mean()
        critic_loss = (advantage ** 2).mean()

        loss = actor_loss + critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (episode + 1) % 40 == 0:
        print(f"Episode {episode + 1:3d} | Final Level: {state:5.1f} (Target 60) | Episode Reward: {sum(rewards):.2f}")
```