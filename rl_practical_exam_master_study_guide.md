# RL Practical Exam — One Template, Twelve Labs

Every single program in this guide follows the **exact same 4-block structure** and uses the **exact same variable names**:
`state, action, next_state, reward, alpha, gamma, epsilon, Q, V, actor, critic`.

Once you memorize the **Master Template** and the **two shared environments** (the 5-State Line World for Tabular RL, and `CartPole-v1` for Deep RL), you only need to swap out **3 to 5 lines of code** inside the `POLICY` and `EQUATION` blocks for any question asked in the exam.

---

## 1. The Master Template (Memorize This Once)

```python
# 1. SETUP                      -> alpha, gamma, epsilon, episodes
# 2. STATES AND ACTIONS         -> environment setup, Q/V tables or actor/critic networks
# 3. TRAINING LOOP
for episode in range(episodes):
    state = start               # (or: state, _ = env.reset())
    while not done:             # (or: while state != goal:)
        # POLICY                -> how to choose action
        # ENVIRONMENT           -> get next_state, reward, done
        # EQUATION              -> the learning update  <-- ONLY THIS CHANGES!
        state = next_state
# 4. OUTPUT                     -> print learned V, Q, or episode rewards
```

---

## 2. Master Cheat Sheet: What Stays the Same vs. What Changes

### A. Which Environment & Data Structure to Use?

| Labs | Category | Environment Used | Memory / Model |
| :--- | :--- | :--- | :--- |
| **Lab 1** | Multi-Armed Bandit | 4-Ad Click Simulator (`true_ctr`) | 1D Array: `Q = np.zeros(n_actions)` |
| **Labs 2, 3, 4** | Tabular Prediction / DP | 5-State Line World (`0->1->2->3->4`) | 1D Array: `V = np.zeros(n_states)` |
| **Labs 5, 6** | Tabular Control | 5-State Line World (`0->1->2->3->4`) | 2D Table: `Q = np.zeros((n_states, n_actions))` |
| **Lab 8** | Self-Play Board RL | 1D Dice-Backgammon (`0..10`) | Neural Net: `critic` ($V(s)$) + trace $\lambda$ |
| **Labs 7, 9, 10, 11, 12** | Deep RL | ` gym.make("CartPole-v1")` | PyTorch `nn.Sequential` (`actor` / `critic`) |

---

### B. The 12 Core Equations (The Only Lines That Change)

1. **LAB-1 (Epsilon-Greedy Ad Bandit):** No states, only actions.
   $$Q(a) \leftarrow Q(a) + \alpha \cdot (r - Q(a))$$
   ```python
   Q[action] += alpha * (reward - Q[action])
   ```

2. **LAB-2 (State Value Estimation / TD(0)):** Bootstraps using the next state's value.
   $$V(s) \leftarrow V(s) + \alpha \cdot (r + \gamma V(s') - V(s))$$
   ```python
   V[state] += alpha * (reward + gamma * V[next_state] - V[state])
   ```

3. **LAB-3 (Policy Iteration):** Exact Bellman expectation backup + greedy action flip (no $\alpha$ needed).
   $$V(s) \leftarrow r + \gamma V(s') \quad \text{then} \quad \pi(s) \leftarrow \arg\max_a (r + \gamma V(s'))$$
   ```python
   V[state] = reward + gamma * V[next_state]
   ```

4. **LAB-4 (Monte Carlo Prediction):** Waits until the episode ends, computes return $G_t$, then updates $V(s)$.
   $$G_t = r_t + \gamma G_{t+1}, \quad V(s_t) \leftarrow V(s_t) + \alpha \cdot (G_t - V(s_t))$$
   ```python
   V[s] += alpha * (G - V[s])
   ```

5. **LAB-5 (Q-Learning — Off-Policy):** Uses $\max_{a'} Q(s', a')$ regardless of what action is actually taken next.
   $$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \left(r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right)$$
   ```python
   Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
   ```

6. **LAB-6 (SARSA — On-Policy):** Chooses `next_action` *before* updating and uses $Q(s', a')$.
   $$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot (r + \gamma Q(s', a') - Q(s, a))$$
   ```python
   Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
   ```

7. **LAB-7 (Deep Q-Learning / DQN):** Same as Q-Learning, but uses a Neural Network and Mean Squared Error loss.
   $$\mathcal{L} = \left(Q_\theta(s, a) - \left[r + \gamma (1 - d)\max_{a'} Q_\theta(s', a')\right]\right)^2$$
   ```python
   target = reward + gamma * (1 - done) * model(next_x).max().detach()
   loss = (model(x)[action] - target) ** 2
   ```

8. **LAB-8 (TD-Gammon Learning):** Neural network value estimation with TD($\lambda$) eligibility traces via self-play.
   $$\delta_t = r + \gamma V_\theta(s') - V_\theta(s), \quad e_t = \gamma \lambda e_{t-1} + \nabla_\theta V_\theta(s)$$
   ```python
   td_error = reward + gamma * (1 - done) * critic(next_x).detach() - critic(x)
   ```

9. **LAB-9 (REINFORCE):** Plays a full episode, calculates discounted returns $G_t$, and scales log-probabilities.
   $$\mathcal{L} = -\frac{1}{T} \sum_{t=0}^{T-1} \log \pi_\theta(a_t | s_t) \cdot G_t$$
   ```python
   loss = -(torch.stack(log_probs) * returns).mean()
   ```

10. **LAB-10 (Actor-Critic — Step-by-Step TD):** Updates *every single step* using TD error as the advantage.
    $$A_t = r + \gamma (1 - d) V_\phi(s') - V_\phi(s), \quad \mathcal{L} = -\log \pi_\theta(a|s) \cdot A_t + A_t^2$$
    ```python
    advantage = reward + gamma * (1 - done) * critic(next_x).detach() - critic(x)
    loss = -dist.log_prob(action) * advantage.detach() + advantage ** 2
    ```

11. **LAB-11 (Actor-Critic Methods — Batch A2C with Entropy):** Updates at the *end of the episode* using Monte Carlo returns $G_t - V(s_t)$ plus an entropy bonus $H(\pi)$ to encourage exploration.
    $$A_t = G_t - V_\phi(s_t), \quad \mathcal{L} = -\log \pi_\theta(a_t|s_t) A_t + A_t^2 - \beta H(\pi_\theta)$$
    ```python
    advantage = returns - critic(xs).squeeze(-1)
    loss = -(log_probs * advantage.detach()).mean() + (advantage ** 2).mean() - 0.01 * entropies.mean()
    ```

12. **LAB-12 (PPO — Proximal Policy Optimization):** Same as Lab 11, but saves `old_log_probs`, runs multiple update epochs, and clips the probability ratio $r_t(\theta) = \exp(\log \pi_\theta - \log \pi_{\theta_{\text{old}}})$ to $[0.8, 1.2]$.
    $$\mathcal{L}^{\text{CLIP}} = -\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 0.8, 1.2) A_t\right) + A_t^2$$
    ```python
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    loss = -torch.min(ratio * advantage.detach(), clipped * advantage.detach()).mean() + (advantage ** 2).mean()
    ```

---

### C. Shared Helper Snippets (Used Across Multiple Labs)

* **Epsilon-Greedy Action Selection (Labs 1, 2, 5, 6, 7):**
  ```python
  if random.random() < epsilon:
      action = random.randint(0, n_actions - 1)   # Explore
  else:
      action = np.argmax(Q[state])                # Exploit
  ```
* **Discounted Returns Calculation (Labs 4, 9, 11, 12):**
  ```python
  G = 0
  returns = []
  for r in reversed(rewards):
      G = r + gamma * G
      returns.insert(0, G)
  ```
* **Emergency Fallback (If `gymnasium` is not installed on your exam PC for Labs 7, 9, 10, 11, 12):**
  If `import gymnasium as gym` throws an error in the exam hall, replace `env = gym.make("CartPole-v1")` with this 8-line dummy CartPole class and **keep the rest of your code 100% identical**:
  ```python
  class DummyCartPole:
      def reset(self):
          self.s = [0.0, 0.0, 0.05, 0.0]; self.steps = 0; return self.s, {}
      def step(self, a):
          self.steps += 1
          self.s[2] += (-0.02 if a == 1 else 0.02) + random.uniform(-0.01, 0.01)
          done = abs(self.s[2]) > 0.2 or self.steps >= 200
          return list(self.s), 1.0, done, False, {}
      def close(self): pass
  env = DummyCartPole()
  ```

---

## LAB-1: Epsilon-Greedy for Choosing Advertisement

### Theory & Intuition
* **Problem:** An online platform has $K$ different advertisements with unknown Click-Through Rates (CTRs). Every time an ad $a$ is shown to a user, it yields a reward $r = 1$ (clicked) or $r = 0$ (ignored).
* **Multi-Armed Bandit:** Unlike full RL, there are **no states** (or only a single stateless context). We only estimate action-values $Q(a)$, representing the expected click rate of each ad.
* **Exploration vs. Exploitation:**
  * With probability $\epsilon$, the agent **explores** by displaying a random ad to gather data.
  * With probability $1 - \epsilon$, the agent **exploits** by picking the ad with the highest estimated click rate: $a = \arg\max_a Q(a)$.
* **Update Rule:** Incremental sample-average update:
  $$Q(a) \leftarrow Q(a) + \alpha \cdot (r - Q(a))$$

### What to Memorize
* `Q` is a 1D array of size `n_actions` (number of ads).
* There is no `next_state` or `gamma` because one ad impression does not transition the user into a spatial board state.

### Code
```python
import numpy as np
import random

# 1. SETUP
alpha = 0.1
epsilon = 0.2
episodes = 1000

# 2. STATES AND ACTIONS
true_ctr = [0.10, 0.50, 0.30, 0.20]                     # hidden click probabilities of 4 Ads
n_actions = len(true_ctr)
Q = np.zeros(n_actions)                                 # estimated CTR for each Ad
impressions = np.zeros(n_actions)

# 3. TRAINING LOOP
for episode in range(episodes):
    # POLICY: epsilon-greedy
    if random.random() < epsilon:
        action = random.randint(0, n_actions - 1)       # explore random Ad
    else:
        action = np.argmax(Q)                           # exploit best Ad

    # ENVIRONMENT: user clicks (1) with probability true_ctr[action], else (0)
    reward = 1 if random.random() < true_ctr[action] else 0
    impressions[action] += 1

    # EQUATION: Bandit update (Q-Learning without next_state)
    Q[action] += alpha * (reward - Q[action])

# 4. OUTPUT
print("Estimated CTR (Q):", np.round(Q, 2))
print("True CTR:         ", true_ctr)
print("Ad Impressions:   ", impressions.astype(int))
print("Best Ad to show:   Ad", np.argmax(Q))
```

---

## LAB-2: State Value Function Estimation (TD(0) Prediction)

### Theory & Intuition
* **Goal:** Estimate the state-value function $V^\pi(s)$—the expected cumulative discounted reward starting from state $s$ and following policy $\pi$.
* **Temporal Difference TD(0):** Instead of waiting until the end of the episode like Monte Carlo, TD(0) updates $V(s)$ **at every single step** using **bootstrapping** (estimating the remaining return using the current estimate of the next state, $V(s')$).
* **TD Target:** $r + \gamma V(s')$
* **TD Error ($\delta$):** $\delta = r + \gamma V(s') - V(s)$
* **Update Rule:**
  $$V(s) \leftarrow V(s) + \alpha \cdot \left(r + \gamma V(s') - V(s)\right)$$

### What Changes from Lab 1?
* We introduce the **5-State Line World** (`0 - 1 - 2 - 3 - 4(Goal)`).
* We track `V = np.zeros(n_states)` instead of `Q`, and add `gamma * V[next_state]` to the equation.

### Code
```python
import numpy as np
import random

# 1. SETUP
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 200

# 2. STATES AND ACTIONS
n_states = 5                                            # 0 - 1 - 2 - 3 - 4 (Goal)
goal = 4
moves = [-1, 1]                                         # 0 = left (-1), 1 = right (+1)
n_actions = len(moves)
V = np.zeros(n_states)

# 3. TRAINING LOOP
for episode in range(episodes):
    state = 0
    while state != goal:
        # POLICY: epsilon-greedy (mostly move right toward goal)
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = 1                                  # move right

        # ENVIRONMENT
        next_state = max(0, min(goal, state + moves[action]))
        reward = 10 if next_state == goal else -1

        # EQUATION: TD(0) State Value Estimation
        V[state] += alpha * (
            reward + gamma * V[next_state] - V[state]
        )

        state = next_state

# 4. OUTPUT
print("Estimated State Values V(s):", np.round(V, 2))
```

---

## LAB-3: Policy Iteration (Dynamic Programming)

### Theory & Intuition
* **Model-Based RL:** Policy Iteration finds the optimal policy $\pi^*$ when the environment's transition rules are known.
* It alternates between two steps until the policy stops changing:
  1. **Policy Evaluation:** Repeatedly sweep through all states and compute the value $V^\pi(s)$ of the current policy $\pi(s)$:
     $$V(s) \leftarrow r(s, \pi(s)) + \gamma V(s')$$
  2. **Policy Improvement:** Make the policy greedy with respect to the updated value table $V$:
     $$\pi(s) \leftarrow \arg\max_a \left[ r(s, a) + \gamma V(s') \right]$$

### What Changes from Lab 2?
* Same 5-State Line World and same `V = np.zeros(n_states)`.
* We also maintain `policy = np.zeros(n_states, dtype=int)`.
* Instead of random episodes, we loop through `for state in range(goal):` to evaluate the current policy, and then pick the best action for each state.

### Code
```python
import numpy as np

# 1. SETUP
gamma = 0.9
iterations = 20

# 2. STATES AND ACTIONS
n_states = 5                                            # 0 - 1 - 2 - 3 - 4 (Goal)
goal = 4
moves = [-1, 1]                                         # 0 = left (-1), 1 = right (+1)
n_actions = len(moves)
V = np.zeros(n_states)
policy = np.zeros(n_states, dtype=int)                  # start with terrible policy: all Left (0)

# 3. TRAINING LOOP (Policy Evaluation + Policy Improvement)
for iteration in range(iterations):
    # STEP 3a: POLICY EVALUATION (compute V under current policy)
    for _ in range(10):
        for state in range(goal):
            action = policy[state]                      # POLICY: read from policy table

            # ENVIRONMENT
            next_state = max(0, min(goal, state + moves[action]))
            reward = 10 if next_state == goal else -1

            # EQUATION: Bellman Expectation Update
            V[state] = reward + gamma * V[next_state]

    # STEP 3b: POLICY IMPROVEMENT (update policy greedily using V)
    for state in range(goal):
        action_values = []
        for action in range(n_actions):
            next_state = max(0, min(goal, state + moves[action]))
            reward = 10 if next_state == goal else -1
            action_values.append(reward + gamma * V[next_state])

        policy[state] = np.argmax(action_values)        # pick best action

# 4. OUTPUT
print("Optimal State Values V*(s):", np.round(V, 2))
print("Optimal Policy (0=Left, 1=Right):", policy[:goal])
```

---

## LAB-4: Monte Carlo Prediction

### Theory & Intuition
* **Goal:** Estimate $V^\pi(s)$ from experience without knowing the environment transitions and **without bootstrapping**.
* **How it Works:** The agent plays an **entire episode** from start to termination, recording all `(state, reward)` pairs.
* Once the episode finishes, it loops **backward** from the final step to compute the true discounted return $G_t$ following each state:
  $$G_t = r_{t} + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots = r_t + \gamma G_{t+1}$$
* Finally, it updates the value of visited states toward the actual observed return $G_t$:
  $$V(s_t) \leftarrow V(s_t) + \alpha \cdot (G_t - V(s_t))$$
* **Key Contrast with TD(0) (Lab 2):** TD(0) updates *inside* the `while` loop using an estimate $r + \gamma V(s')$. Monte Carlo updates *after* the `while` loop using the real return $G_t$.

### What Changes from Lab 2?
* Inside the `while state != goal:` loop, we just append `(state, reward)` to `history`.
* The `EQUATION` block moves **outside** the `while` loop to compute `G = reward + gamma * G` backward.

### Code
```python
import numpy as np
import random

# 1. SETUP
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 200

# 2. STATES AND ACTIONS
n_states = 5                                            # 0 - 1 - 2 - 3 - 4 (Goal)
goal = 4
moves = [-1, 1]                                         # 0 = left (-1), 1 = right (+1)
n_actions = len(moves)
V = np.zeros(n_states)

# 3. TRAINING LOOP
for episode in range(episodes):
    state = 0
    history = []                                        # stores (state, reward) for the episode

    while state != goal:
        # POLICY: epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = 1                                  # move right

        # ENVIRONMENT
        next_state = max(0, min(goal, state + moves[action]))
        reward = 10 if next_state == goal else -1

        history.append((state, reward))
        state = next_state

    # EQUATION: Monte Carlo Prediction (runs AFTER episode ends)
    G = 0
    for s, reward in reversed(history):
        G = reward + gamma * G
        V[s] += alpha * (G - V[s])

# 4. OUTPUT
print("Monte Carlo State Values V(s):", np.round(V, 2))
```

---

## LAB-5: Q-Learning (Off-Policy TD Control)

### Theory & Intuition
* **From Prediction to Control:** Labs 2 and 4 only evaluated $V(s)$ for a fixed policy. To *learn* the optimal actions without a model of the environment, we learn the action-value table $Q(s, a)$.
* **Off-Policy Learning:** Q-Learning is **off-policy** because its behavior policy (how it acts) is $\epsilon$-greedy, but its target policy (how it updates) is **100% greedy** ($\max_{a'} Q(s', a')$). Even if the agent explores randomly on the next step, Q-Learning updates as if the agent will take the best possible action in $s'$.
* **Update Rule:**
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \left(r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right)$$

### What Changes from Lab 2?
* `V = np.zeros(n_states)` becomes a 2D table: `Q = np.zeros((n_states, n_actions))`.
* In `POLICY`, `action = np.argmax(Q[state])`.
* In `EQUATION`, `V[next_state]` becomes `np.max(Q[next_state])`.

### Code
```python
import numpy as np
import random

# 1. SETUP
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

# 2. STATES AND ACTIONS
n_states = 5                                            # 0 - 1 - 2 - 3 - 4 (Goal)
goal = 4
moves = [-1, 1]                                         # 0 = left (-1), 1 = right (+1)
n_actions = len(moves)
Q = np.zeros((n_states, n_actions))

# 3. TRAINING LOOP
for episode in range(episodes):
    state = 0
    while state != goal:
        # POLICY: epsilon-greedy on Q
        if random.random() < epsilon:
            action = random.randint(0, n_actions - 1)
        else:
            action = np.argmax(Q[state])

        # ENVIRONMENT
        next_state = max(0, min(goal, state + moves[action]))
        reward = 10 if next_state == goal else -1

        # EQUATION: Q-Learning (uses np.max)
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

# 4. OUTPUT
print("Learned Q-Table:\n", np.round(Q, 2))
print("Optimal Action per State (0=Left, 1=Right):", np.argmax(Q[:goal], axis=1))
```

---

## LAB-6: SARSA Implementation (On-Policy TD Control)

### Theory & Intuition
* **Name Origin:** **S**tate ($s$), **A**ction ($a$), **R**eward ($r$), next **S**tate ($s'$), next **A**ction ($a'$).
* **On-Policy Learning:** Unlike Q-Learning (which optimistically assumes the best action $\max_{a'} Q(s', a')$ will be taken next), SARSA is **on-policy**. It selects the actual `next_action` $a'$ using the $\epsilon$-greedy policy *before* performing the update, and uses $Q(s', a')$ in the target.
* Because it accounts for its own exploratory mistakes ($\epsilon$), SARSA learns safer paths near hazards (like in the Cliff Walking problem).
* **Update Rule:**
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \cdot \left(r + \gamma Q(s', a') - Q(s, a)\right)$$

### What Changes from Lab 5 (Q-Learning)?
Only **3 small changes** (marked `# SARSA` below):
1. Pick the first `action` *before* the `while` loop starts.
2. Pick `next_action` *inside* the `while` loop after observing `next_state`.
3. Replace `np.max(Q[next_state])` with `Q[next_state, next_action]`, and set `action = next_action`.

### Code
```python
import numpy as np
import random

# 1. SETUP
alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

# 2. STATES AND ACTIONS
n_states = 5                                            # 0 - 1 - 2 - 3 - 4 (Goal)
goal = 4
moves = [-1, 1]                                         # 0 = left (-1), 1 = right (+1)
n_actions = len(moves)
Q = np.zeros((n_states, n_actions))

# 3. TRAINING LOOP
for episode in range(episodes):
    state = 0

    # POLICY: first action                              # SARSA 1: choose action before loop
    if random.random() < epsilon:
        action = random.randint(0, n_actions - 1)
    else:
        action = np.argmax(Q[state])

    while state != goal:
        # ENVIRONMENT
        next_state = max(0, min(goal, state + moves[action]))
        reward = 10 if next_state == goal else -1

        # POLICY: next action                           # SARSA 2: choose next_action now
        if random.random() < epsilon:
            next_action = random.randint(0, n_actions - 1)
        else:
            next_action = np.argmax(Q[next_state])

        # EQUATION: SARSA                               # SARSA 3: use Q[next_state, next_action]
        Q[state, action] += alpha * (
            reward + gamma * Q[next_state, next_action] - Q[state, action]
        )

        state = next_state
        action = next_action

# 4. OUTPUT
print("Learned SARSA Q-Table:\n", np.round(Q, 2))
print("Optimal Action per State (0=Left, 1=Right):", np.argmax(Q[:goal], axis=1))
```

---

## LAB-7: Deep Q-Learning (DQN) for CartPole

### Theory & Intuition
* **Why Deep Q-Learning?** In `CartPole-v1`, the state consists of 4 continuous numbers (cart position, cart velocity, pole angle, pole angular velocity). Because continuous states have infinitely many values, we cannot build a tabular `Q[state, action]` matrix.
* **Function Approximation:** We replace the Q-table with a Neural Network $Q_\theta(s)$ that takes the 4 state numbers as input and outputs 2 Q-values (push left, push right).
* **Experience Replay:** Standard online updates correlate consecutive states. Storing transitions `(x, action, reward, next_x, done)` in a replay buffer (`deque`) and sampling random mini-batches stabilizes training.
* **Loss Function:** Mean Squared Bellman Error:
  $$\mathcal{L}(\theta) = \left( Q_\theta(s, a) - \left[ r + \gamma (1 - d) \max_{a'} Q_\theta(s', a') \right] \right)^2$$

### What Changes from Lab 5 (Q-Learning)?
* `Q = np.zeros(...)` is replaced by `model = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))`.
* Instead of `Q[state, action] += ...`, we compute `loss = (q_value - target) ** 2` and run `optimizer.zero_grad(); loss.backward(); optimizer.step()`.

### Code
```python
import gymnasium as gym
import torch
import torch.nn as nn
import random
from collections import deque

# 1. SETUP
gamma = 0.99
epsilon = 0.2
episodes = 150
memory = deque(maxlen=2000)                             # Experience Replay Buffer

# 2. STATES AND ACTIONS
env = gym.make("CartPole-v1")                           # 4 state inputs, 2 actions
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

        # EQUATION: Store in replay memory & train on random batch
        memory.append((x, action, reward, next_x, float(done)))

        if len(memory) >= 32:
            batch = random.sample(memory, 32)
            loss = 0
            for bx, ba, br, bnext_x, bdone in batch:
                q_value = model(bx)[ba]
                target = br + gamma * (1 - bdone) * model(bnext_x).max().detach()
                loss += (q_value - target) ** 2

            loss = loss / 32
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state

    if (episode + 1) % 30 == 0:
        print("Episode", episode + 1, "Reward:", total_reward)

# 4. OUTPUT
env.close()
```

---

## LAB-8: TD-Gammon Learning

### Theory & Intuition
* **What is TD-Gammon?** Developed by Gerald Tesauro (1992) for Backgammon, TD-Gammon was one of the first breakthroughs in neural reinforcement learning. Because Backgammon has roughly $10^{20}$ board states and dice rolls make transitions stochastic, tabular RL is impossible.
* **Three Core Pillars of TD-Gammon:**
  1. **Afterstate Value Network $V_\theta(s)$:** A neural network (`critic`) takes the board configuration and outputs a single scalar predicting the probability of winning from that board state.
  2. **Self-Play:** The agent plays against itself, choosing moves that maximize $V_\theta(s_{\text{next}})$ after rolling the dice.
  3. **TD($\lambda$) with Eligibility Traces:** Instead of only updating the current board state, TD-Gammon maintains an **eligibility trace** $e_t$ for network weights, decaying by $\gamma \lambda$ each turn so earlier board states rapidly receive credit when a player wins:
     $$\delta_t = r_t + \gamma V_\theta(s_{t+1}) - V_\theta(s_t)$$
     $$e_t = \gamma \lambda e_{t-1} + \nabla_\theta V_\theta(s_t), \quad \theta \leftarrow \theta + \alpha \delta_t e_t$$

### How We Keep the Code Simple & Exam-Ready
* We implement a simplified **1D Backgammon Dice-Bearing-Off Game** (board positions `0` to `10`, roll dice `1..3`, choose between aggressive leap vs. safe step) using our exact `critic` network and a 4-line PyTorch TD($\lambda$) trace update.

### Code
```python
import torch
import torch.nn as nn
import random

# 1. SETUP
alpha = 0.01
gamma = 0.95
lam = 0.7                                               # TD(lambda) eligibility decay parameter
episodes = 200

# 2. STATES AND ACTIONS
goal = 10                                               # simplified Backgammon bear-off point
critic = nn.Sequential(                                 # TD-Gammon Board Evaluation Network V(s)
    nn.Linear(1, 16), nn.ReLU(),
    nn.Linear(16, 1), nn.Sigmoid()                      # outputs win probability in [0, 1]
)

# 3. TRAINING LOOP (Self-Play + TD(lambda) Eligibility Traces)
for episode in range(episodes):
    state = 0                                           # checker starts at position 0
    traces = [torch.zeros_like(p) for p in critic.parameters()]

    while state < goal:
        dice = random.randint(1, 3)                     # roll dice (1, 2, or 3)
        x = torch.tensor([state / goal], dtype=torch.float32)

        # POLICY: evaluate candidate moves using V(next_state) and pick the best
        candidates = [min(goal, state + dice), min(goal, state + 1)]
        best_next = max(candidates, key=lambda s: critic(torch.tensor([s / goal], dtype=torch.float32)).item())

        # ENVIRONMENT
        next_state = best_next
        reward = 1.0 if next_state == goal else 0.0     # 1.0 on win, 0 otherwise
        next_x = torch.tensor([next_state / goal], dtype=torch.float32)

        # EQUATION: TD-Gammon TD(lambda) with Eligibility Traces
        value = critic(x)
        next_value = 0.0 if next_state == goal else critic(next_x).detach()
        td_error = (reward + gamma * next_value - value).detach()

        critic.zero_grad()
        value.backward()                                # computes grad of V(s)

        with torch.no_grad():
            for param, trace in zip(critic.parameters(), traces):
                trace.mul_(gamma * lam).add_(param.grad) # e_t = gamma * lambda * e_{t-1} + grad V(s)
                param.add_(alpha * td_error * trace)     # theta += alpha * td_error * e_t

        state = next_state

# 4. OUTPUT
print("TD-Gammon Learned Win Probabilities V(board_position):")
for pos in [0, 2, 5, 8, 9]:
    prob = critic(torch.tensor([pos / goal], dtype=torch.float32)).item()
    print(f"Position {pos} -> V(s) = {prob:.3f}")
```

---

## LAB-9: CartPole using REINFORCE (Monte Carlo Policy Gradient)

### Theory & Intuition
* **Value-Based vs. Policy-Based:** In Labs 5–7, we learned $Q$-values and derived a policy using $\arg\max$. In **Policy Gradient** methods, we directly parameterize a stochastic policy network $\pi_\theta(a|s)$ (called the `actor`) that outputs action probabilities via `Softmax`.
* **How REINFORCE Works:**
  1. Play a **complete episode** using the `actor` network, storing `log_probs` ($\log \pi_\theta(a_t|s_t)$) and `rewards` ($r_t$) at every step.
  2. Compute the discounted Monte Carlo return $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$ for each step.
  3. Update network weights to increase the probability of actions that led to high returns $G_t$:
     $$\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot G_t \right]$$
* **Loss Function (for Gradient Descent):**
  $$\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=0}^{T-1} \log \pi_\theta(a_t | s_t) \cdot G_t$$

### What Changes from Lab 4 (Monte Carlo)?
* Instead of updating a table `V[s]`, we normalize `returns` and compute `loss = -(torch.stack(log_probs) * returns).mean()`.

### Code
```python
import gymnasium as gym
import torch
import torch.nn as nn

# 1. SETUP
gamma = 0.99
episodes = 200

# 2. STATES AND ACTIONS
env = gym.make("CartPole-v1")                           # 4 state inputs, 2 actions
n_actions = 2
actor = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, n_actions), nn.Softmax(dim=-1)
)
optimizer = torch.optim.Adam(actor.parameters(), lr=0.005)

# 3. TRAINING LOOP
for episode in range(episodes):
    state, _ = env.reset()
    log_probs = []
    rewards = []
    done = False

    while not done:
        x = torch.tensor(state, dtype=torch.float32)

        # POLICY: sample action from actor network probabilities
        dist = torch.distributions.Categorical(actor(x))
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        # ENVIRONMENT
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    # EQUATION: REINFORCE (runs AFTER the episode finishes)
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = -(torch.stack(log_probs) * returns).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (episode + 1) % 40 == 0:
        print("Episode", episode + 1, "Reward:", sum(rewards))

# 4. OUTPUT
env.close()
```

---

## LAB-10: Actor-Critic (1-Step Online TD Actor-Critic)

### Theory & Intuition
* **Why Actor-Critic?** REINFORCE (Lab 9) suffers from **high variance** because full-episode Monte Carlo returns $G_t$ fluctuate wildly depending on random future events, and it must wait until the end of the episode to learn.
* **The Solution:** Combine Policy Gradients (Lab 9) with TD Value Estimation (Lab 2):
  * **Actor ($\pi_\theta(a|s)$):** The policy network that chooses actions.
  * **Critic ($V_\phi(s)$):** The value network that evaluates how good the state is.
* **1-Step TD Advantage:** At **every step**, the Critic computes the TD error (Advantage):
  $$A_t = r + \gamma (1 - d) V_\phi(s') - V_\phi(s)$$
  If $A_t > 0$, the action went better than expected, so the Actor increases its probability.
* **Combined Loss Function:**
  $$\mathcal{L} = \underbrace{-\log \pi_\theta(a|s) \cdot A_t}_{\text{Actor Loss}} + \underbrace{A_t^2}_{\text{Critic Loss}}$$

### What Changes from Lab 9 (REINFORCE)?
* We add a second network: `critic = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 1))`.
* The `EQUATION` moves **inside** the `while not done:` loop (updates every step using `advantage = target - value`).

### Code
```python
import gymnasium as gym
import torch
import torch.nn as nn

# 1. SETUP
gamma = 0.99
episodes = 200

# 2. STATES AND ACTIONS
env = gym.make("CartPole-v1")                           # 4 state inputs, 2 actions
n_actions = 2
actor = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, n_actions), nn.Softmax(dim=-1)
)
critic = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, 1)
)
optimizer = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=0.003
)

# 3. TRAINING LOOP
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        x = torch.tensor(state, dtype=torch.float32)

        # POLICY: actor
        dist = torch.distributions.Categorical(actor(x))
        action = dist.sample()

        # ENVIRONMENT
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        next_x = torch.tensor(next_state, dtype=torch.float32)
        total_reward += reward

        # EQUATION: 1-Step TD Actor-Critic (updates EVERY step)
        value = critic(x).squeeze()
        next_value = critic(next_x).squeeze().detach()
        target = reward + gamma * (1 - float(done)) * next_value
        advantage = target - value

        loss = -dist.log_prob(action) * advantage.detach() + advantage ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    if (episode + 1) % 40 == 0:
        print("Episode", episode + 1, "Reward:", total_reward)

# 4. OUTPUT
env.close()
```

---

## LAB-11: Actor-Critic Methods (Advantage Actor-Critic / A2C)

### Theory & Intuition
* **Why Lab 10 vs. Lab 11?** While basic Actor-Critic (Lab 10) updates step-by-step with 1-step TD error, modern **Advantage Actor-Critic (A2C)** methods collect a batch/episode trajectory and incorporate two key improvements:
  1. **Multi-Step Baseline Advantage:** Uses full rollout returns $G_t$ minus the Critic's baseline $V_\phi(s_t)$:
     $$A_t = G_t - V_\phi(s_t)$$
     Subtracting the baseline $V_\phi(s_t)$ drastically reduces gradient variance without changing the expected gradient.
  2. **Entropy Regularization ($H(\pi)$):** Adds the policy entropy $H(\pi_\theta(s)) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$ to prevent premature convergence to a single suboptimal action:
     $$\mathcal{L} = -\frac{1}{T}\sum_t \log \pi_\theta(a_t|s_t) A_t + \frac{1}{T}\sum_t A_t^2 - \beta H(\pi_\theta)$$

### Why Memorizing This Is a Superpower
* Lab 11 is **REINFORCE (Lab 9) + 3 lines for the Critic**.
* And Lab 12 (PPO) is **Lab 11 + 3 lines for clipping**!

### Code
```python
import gymnasium as gym
import torch
import torch.nn as nn

# 1. SETUP
gamma = 0.99
episodes = 200

# 2. STATES AND ACTIONS
env = gym.make("CartPole-v1")                           # 4 state inputs, 2 actions
n_actions = 2
actor = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, n_actions), nn.Softmax(dim=-1)
)
critic = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, 1)
)
optimizer = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=0.003
)

# 3. TRAINING LOOP
for episode in range(episodes):
    state, _ = env.reset()
    xs, log_probs, entropies, rewards = [], [], [], []
    done = False

    while not done:
        x = torch.tensor(state, dtype=torch.float32)

        # POLICY: actor
        dist = torch.distributions.Categorical(actor(x))
        action = dist.sample()

        xs.append(x)
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        # ENVIRONMENT
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    # EQUATION: Advantage Actor-Critic (A2C) with Entropy Bonus
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    xs = torch.stack(xs)
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    values = critic(xs).squeeze(-1)
    advantage = returns - values                        # A_t = G_t - V(s_t)

    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = (advantage ** 2).mean()
    entropy_bonus = entropies.mean()

    loss = actor_loss + critic_loss - 0.01 * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (episode + 1) % 40 == 0:
        print("Episode", episode + 1, "Reward:", sum(rewards))

# 4. OUTPUT
env.close()
```

---

## LAB-12: Proximal Policy Optimization (PPO)

### Theory & Intuition
* **The Problem with Standard Policy Gradients:** In REINFORCE (Lab 9) and A2C (Lab 11), we can only do **one gradient step** per episode because taking multiple gradient steps on the same trajectory pushes the new policy $\pi_\theta$ too far away from the policy that collected the data ($\pi_{\theta_{\text{old}}}$), causing catastrophic performance collapse.
* **How PPO Solves This:**
  1. **Importance Sampling Ratio ($r_t(\theta)$):** Measures how much the new policy differs from the old policy:
     $$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} = \exp\left(\log \pi_\theta(a_t | s_t) - \log \pi_{\theta_{\text{old}}}(a_t | s_t)\right)$$
  2. **Clipped Surrogate Objective:** Clips $r_t(\theta)$ into the safe interval $[1 - \epsilon, 1 + \epsilon]$ (typically $[0.8, 1.2]$ for $\epsilon = 0.2$) and takes the minimum so the policy never changes too drastically in a single update:
     $$\mathcal{L}^{\text{CLIP}}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) A_t \right) \right]$$

### What Changes from Lab 11 (A2C)?
Only **3 additions** (marked `# PPO 1`, `# PPO 2`, `# PPO 3` in the code):
1. Save `old_log_probs.append(dist.log_prob(action).detach())` during the episode.
2. Wrap the update in a `for _ in range(4):` loop to reuse the same trajectory multiple times.
3. Compute `ratio = torch.exp(new_log_probs - old_log_probs)` and clip it with `torch.clamp(ratio, 0.8, 1.2)`.

### Code
```python
import gymnasium as gym
import torch
import torch.nn as nn

# 1. SETUP
gamma = 0.99
episodes = 200

# 2. STATES AND ACTIONS
env = gym.make("CartPole-v1")                           # 4 state inputs, 2 actions
n_actions = 2
actor = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, n_actions), nn.Softmax(dim=-1)
)
critic = nn.Sequential(
    nn.Linear(4, 32), nn.ReLU(),
    nn.Linear(32, 1)
)
optimizer = torch.optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), lr=0.003
)

# 3. TRAINING LOOP
for episode in range(episodes):
    state, _ = env.reset()
    xs, actions, old_log_probs, rewards = [], [], [], []
    done = False

    while not done:
        x = torch.tensor(state, dtype=torch.float32)

        # POLICY: actor
        dist = torch.distributions.Categorical(actor(x))
        action = dist.sample()

        xs.append(x)
        actions.append(action)
        old_log_probs.append(dist.log_prob(action).detach())    # PPO 1: save old log_prob

        # ENVIRONMENT
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        done = terminated or truncated

    # EQUATION: PPO (Clipped Surrogate Objective)
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    xs = torch.stack(xs)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs)
    returns = torch.tensor(returns, dtype=torch.float32)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    for _ in range(4):                                          # PPO 2: multiple update epochs
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
        print("Episode", episode + 1, "Reward:", sum(rewards))

# 4. OUTPUT
env.close()
```

---

## 3. Last-Minute 5-Minute Revision Map

Study the programs in **4 connected families** instead of 12 isolated scripts:

### Family 1: The Tabular $V(s)$ Family (Labs 2, 3, 4)
* All three use `V = np.zeros(5)` on the `0 - 1 - 2 - 3 - 4` Line World.
* **Lab 2 (TD(0)):** Updates *inside* the step loop: `V[s] += alpha * (r + gamma * V[s'] - V[s])`.
* **Lab 3 (Policy Iteration):** No $\alpha$, sweeps `for state in range(goal): V[s] = r + gamma * V[s']`, then updates `policy[s] = np.argmax(...)`.
* **Lab 4 (Monte Carlo):** Updates *after* the episode ends using backward return `G = r + gamma * G`, then `V[s] += alpha * (G - V[s])`.

### Family 2: The Tabular $Q(s, a)$ Family (Labs 1, 5, 6)
* All three use $\epsilon$-greedy action selection (`if random.random() < epsilon: ... else: np.argmax(...)`).
* **Lab 1 (Ad Bandit):** `Q[a] += alpha * (r - Q[a])` (no states, no $\gamma$).
* **Lab 5 (Q-Learning):** `Q[s, a] += alpha * (r + gamma * np.max(Q[s']) - Q[s, a])` (Off-policy).
* **Lab 6 (SARSA):** `Q[s, a] += alpha * (r + gamma * Q[s', a'] - Q[s, a])` (On-policy, pick `next_action` before updating).

### Family 3: Value-Based Neural RL (Labs 7, 8)
* **Lab 7 (DQN):** Q-Learning where `Q[s, a]` is replaced by `model(x)[action]` and updated via squared error `(q_value - target) ** 2` using a `deque` replay buffer.
* **Lab 8 (TD-Gammon):** TD($\lambda$) where `V[s]` is replaced by `critic(x)` and weights are updated with eligibility traces `trace.mul_(gamma * lam).add_(param.grad)`.

### Family 4: Policy Gradient & Actor-Critic Family (Labs 9, 10, 11, 12)
* All four use `dist = Categorical(actor(x)); action = dist.sample()` on `CartPole-v1`.
* **Lab 9 (REINFORCE):** End of episode $\rightarrow$ compute `returns` $\rightarrow$ `loss = -(log_probs * returns).mean()`.
* **Lab 10 (1-Step Actor-Critic):** Every step $\rightarrow$ `adv = r + gamma * critic(next_x) - critic(x)` $\rightarrow$ `loss = -log_prob * adv.detach() + adv ** 2`.
* **Lab 11 (A2C):** End of episode $\rightarrow$ `adv = returns - critic(xs)` $\rightarrow$ `loss = -(log_probs * adv.detach()).mean() + (adv ** 2).mean() - 0.01 * entropies.mean()`.
* **Lab 12 (PPO):** Same as Lab 11, but run `for _ in range(4):` with `ratio = exp(new_log_probs - old_log_probs)` and `min(ratio * adv, clamp(ratio, 0.8, 1.2) * adv)`.