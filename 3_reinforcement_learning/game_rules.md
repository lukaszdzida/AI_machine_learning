In this activity, you will use a simple grid environment (5 × 5) in which the agent starts at a random position and must navigate to a goal state while avoiding pitfalls. The environment includes:

- States: each cell on the grid is a unique state.

- Actions: the agent can move up, down, left, or right.

- Rewards:
  - +10 for reaching the goal state (position 24).
  - –10 for falling into a pit (position 12).
  - –1 for all other movements (to encourage faster goal-reaching).

The objective is to compare how each algorithm—Q-learning and policy gradients—handles this environment and analyze their behavior.
