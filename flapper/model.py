import numpy
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio
import matplotlib.path as mpath
from matplotlib.markers import MarkerStyle
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import matplotlib.colors as mcolors
import os
import re
import pandas as pd
from .env import InitialCondition
from .env import MultiSwimmerEnv
from mpl_toolkits.mplot3d import Axes3D
import csv
from tqdm import tqdm

plt.ioff() 

if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     print("MPS is available. Using GPU.")
#     device = torch.device("mps")
else:
    print("GPU is not available. Using CPU.")
    device = torch.device("cpu")

class Memory(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                nn.Softmax(dim=-1)
                )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))

        return action.item()
    
    def act_test(self, state):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        action = torch.argmax(action_probs)

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO(object):
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = rewards.type(torch.float32)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Ensure state_values are float32
            state_values = state_values.type(torch.float32)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


class MultiAgentTrainer(object):
    """
    Trains a chain of followers in `MultiSwimmerEnv` with independent PPO policies (one per follower).
    Default n_followers=1 behaves like the single follower conceptually, but training is per-agent.
    """
    def __init__(self, n_followers=1, **env_args):
        # Environment
        self.env_name = "FlappersChain"
        self.env = MultiSwimmerEnv(n_followers=n_followers, **env_args)

        # Per-agent dims
        self.n_agents = self.env.n_followers
        self.state_dim = self.env.observation_space_per_agent.shape[0]
        self.action_dim = 4

        # Hyperparameters
        self.render = False
        self.log_interval = 20
        self.max_episodes = 1000
        self.max_timesteps = 500
        self.n_latent_var = 128
        self.update_timestep = 200
        self.lr = 0.001
        self.betas = (0.9, 0.999)
        self.gamma = 0.9
        self.K_epochs = 4
        self.eps_clip = 0.05

        # Create one PPO + Memory per agent
        self.memories = [Memory() for _ in range(self.n_agents)]
        self.agents = [
            PPO(self.state_dim, self.action_dim, self.n_latent_var, self.lr, self.betas, self.gamma, self.K_epochs, self.eps_clip)
            for _ in range(self.n_agents)
        ]
        # Sequential-training mode flag
        self.sequential = False

    def _normalize_agent_index_input(self, agent_identifier):
        """Accept 1-based agent id (1..n) and return 0-based index.
        Raises ValueError if out of range or not an int.
        """
        if isinstance(agent_identifier, int) and 1 <= agent_identifier <= self.n_agents:
            return agent_identifier - 1
        raise ValueError(
            f"agent_idx must be 1..{self.n_agents} (received {agent_identifier})"
        )

    def _policy_filename(self, agent_idx, episodes):
        # agent_idx is 0-based internally; filenames use 1-based AG ids
        # Canonical order: REW -> noise -> episodes
        if getattr(self, 'sequential', False):
            # In sequential mode, save under Policies/ and replace N{n} with SEQ
            return os.path.join(
                "Policies",
                f"PPO_{self.env_name}_SEQ_AG{agent_idx+1}_OBS_{self.env.observations}_REW_{self.env.rewards}_noise_{self.env.noise}_{episodes}.pth"
            )
        # Non-sequential: save into project root
        return f"./PPO_{self.env_name}_N{self.n_agents}_AG{agent_idx+1}_OBS_{self.env.observations}_REW_{self.env.rewards}_noise_{self.env.noise}_{episodes}.pth"

    def _log_filename(self, agent_idx):
        """Return full path to log CSV for the given agent, using SEQ vs N{n} naming based on mode."""
        if getattr(self, 'sequential', False):
            return os.path.join(
                "Logs",
                f"log_PPO_{self.env_name}_SEQ_AG{agent_idx+1}_OBS_{self.env.observations}_REW_{self.env.rewards}_noise_{self.env.noise}.csv"
            )
        return os.path.join(
            "Logs",
            f"log_PPO_{self.env_name}_N{self.n_agents}_AG{agent_idx+1}_OBS_{self.env.observations}_REW_{self.env.rewards}_noise_{self.env.noise}.csv"
        )

    def load_models(self, episodes=None, map_location=None, clone=None, path=None):
        """Load policies for all agents.

        - If episodes is None, selects the latest checkpoint per agent, honoring the
          canonical filename pattern (REW -> noise -> episodes) and searching the
          appropriate directories based on sequential vs non-sequential naming.
        - If episodes is provided, attempts both naming schemes (sequential and
          non-sequential) for robustness, and searches in both "." and "Policies".
        - Uses map_location to ensure compatibility across CPU/GPU.
        - If clone is an integer (1-based agent id), load only that agent's policy
          (respecting the same episode selection rules) and apply it to all agents.
        """
        # Default to the globally selected device
        if map_location is None:
            map_location = device

        # If clone mode is requested, load the specified agent once and copy to all
        if clone is not None:
            if path is not None:
                state_dict = torch.load(path, map_location=map_location)
                for i in range(self.n_agents):
                    self.agents[i].policy.load_state_dict(state_dict)
                    self.agents[i].policy_old.load_state_dict(state_dict)
                print(f"Cloned policy from agent {clone} to all agents from: {path}")
                return
            src_idx = self._normalize_agent_index_input(clone)
            original_sequential = bool(getattr(self, 'sequential', False))

            candidate_paths = []

            def collect_candidates_for_agent(agent_zero_based, sequential_flag):
                prev = getattr(self, 'sequential', False)
                self.sequential = sequential_flag
                try:
                    if episodes is None:
                        basename = self.get_policy(agent_zero_based + 1, episodes=None)
                    else:
                        basename = os.path.basename(self._policy_filename(agent_zero_based, episodes))
                except FileNotFoundError:
                    basename = None
                finally:
                    self.sequential = prev
                if basename is None:
                    return
                dirs = ["Policies"] if sequential_flag else [".", "Policies"]
                for d in dirs:
                    candidate_paths.append(os.path.join(d, basename))

            collect_candidates_for_agent(src_idx, original_sequential)
            collect_candidates_for_agent(src_idx, not original_sequential)

            seen = set()
            candidates = [p for p in candidate_paths if not (p in seen or seen.add(p))]
            path = next((p for p in candidates if os.path.exists(p)), None)
            if path is None:
                raise FileNotFoundError(f"Model file not found for clone agent {src_idx+1}. Checked: {candidates}")

            state_dict = torch.load(path, map_location=map_location)
            for i in range(self.n_agents):
                self.agents[i].policy.load_state_dict(state_dict)
                self.agents[i].policy_old.load_state_dict(state_dict)
            print(f"Cloned policy from agent {src_idx+1} to all agents from: {path}")
            return

        for i in range(self.n_agents):
            original_sequential = bool(getattr(self, 'sequential', False))

            candidate_paths = []

            def collect_candidates(sequential_flag):
                # Temporarily set sequential flag to build the correct basename/search
                prev = getattr(self, 'sequential', False)
                self.sequential = sequential_flag
                try:
                    if episodes is None:
                        # Let helper locate the best (latest) basename for this agent
                        basename = self.get_policy(i + 1, episodes=None)
                    else:
                        # Construct canonical basename for the requested episode
                        basename = os.path.basename(self._policy_filename(i, episodes))
                except FileNotFoundError:
                    basename = None
                finally:
                    # Restore the previous flag before building directories
                    self.sequential = prev

                if basename is None:
                    return

                # Determine search directories for this naming scheme
                dirs = ["Policies"] if sequential_flag else [".", "Policies"]
                for d in dirs:
                    candidate_paths.append(os.path.join(d, basename))

            # Try with current naming scheme, then the alternate for backwards-compat
            collect_candidates(original_sequential)
            collect_candidates(not original_sequential)

            # De-duplicate while preserving order
            seen = set()
            candidates = [p for p in candidate_paths if not (p in seen or seen.add(p))]
            if path is not None:
                state_dict = torch.load(path, map_location=map_location)
                for i in range(self.n_agents):
                    self.agents[i].policy.load_state_dict(state_dict)
                    self.agents[i].policy_old.load_state_dict(state_dict)
                print(f"Loaded policy for all agents from: {path}")
                return
            
            path = next((p for p in candidates if os.path.exists(p)), None)
            if path is None:
                raise FileNotFoundError(f"Model file not found. Checked: {candidates}")

            state_dict = torch.load(path, map_location=map_location)
            self.agents[i].policy.load_state_dict(state_dict)
            self.agents[i].policy_old.load_state_dict(state_dict)
            print(f"Loaded policy for agent {i+1} from: {path}")

    def train(self, initial_condition=None, initial_condition_fn=None, episodes=None, sequential=False, dynamic=False):
        running_rewards = [0.0 for _ in range(self.n_agents)]
        avg_length = 0
        timestep = 0

        # Configure sequential mode for naming and lookup
        self.sequential = bool(sequential)

        # If dynamic early-stopping is enabled and episodes is None, allow unbounded training
        if episodes is None and not dynamic:
            episodes = self.max_episodes

        # Headers per agent: create Logs dir and only create a new log if it doesn't already exist
        if not os.path.exists("Logs"):
            os.makedirs("Logs")
        for i in range(self.n_agents):
            log_path = self._log_filename(i)
            if not os.path.exists(log_path):
                with open(log_path, 'w') as f:
                    f.write('Episode,avg length,reward\n')

        # Early stopping params: use batches aligned with logging interval
        TARGET_AVG = -10.0
        REQUIRED_BATCHES = 5
        effective_batch = self.log_interval if (hasattr(self, 'log_interval') and self.log_interval and self.log_interval > 0) else 20

        if not sequential:
            # Track early-stopping per agent
            consec_good_batches = [0 for _ in range(self.n_agents)]
            active_train = [True for _ in range(self.n_agents)]
            batch_rewards = [0.0 for _ in range(self.n_agents)]

            # Episode counter; may be unbounded if dynamic and episodes is None
            i_episode = 0
            while True:
                i_episode += 1
                if (not dynamic) and episodes is not None and i_episode > episodes:
                    break
                states = self.env.reset(initial_condition=initial_condition, initial_condition_fn=initial_condition_fn)
                # Ensure we have per-agent states
                assert len(states) == self.n_agents

                # Per-episode reward accumulator for dynamic batches
                ep_rewards = [0.0 for _ in range(self.n_agents)]

                for t in range(self.max_timesteps):
                    timestep += 1
                    # Select actions per agent
                    actions = []
                    for i in range(self.n_agents):
                        if active_train[i]:
                            action = self.agents[i].policy_old.act(states[i], self.memories[i])
                        else:
                            # Frozen agent acts greedily; do not collect memory
                            action = self.agents[i].policy.act_test(states[i])
                        actions.append(action)

                    # Environment step
                    next_states, rewards, done, info = self.env.step(actions)

                    # Store rewards and terminal flags per agent
                    for i in range(self.n_agents):
                        if active_train[i]:
                            self.memories[i].rewards.append(float(rewards[i]))
                            self.memories[i].is_terminals.append(done)
                            running_rewards[i] += float(rewards[i])
                        # Track per-episode reward for dynamic batches
                        ep_rewards[i] += float(rewards[i])

                    # Update all policies periodically
                    if timestep % self.update_timestep == 0:
                        for i in range(self.n_agents):
                            if active_train[i]:
                                self.agents[i].update(self.memories[i])
                                self.memories[i].clear_memory()
                        timestep = 0

                    if self.render:
                        self.env.render()
                    if done:
                        break

                    states = next_states
                    avg_length += 1

                # After episode ends, add per-episode reward to batch accumulators
                for i in range(self.n_agents):
                    if active_train[i]:
                        batch_rewards[i] += ep_rewards[i]

                # Save checkpoints occasionally
                if i_episode % 1000 == 0:
                    for i in range(self.n_agents):
                        torch.save(self.agents[i].policy.state_dict(), self._policy_filename(i, i_episode))

                # Logging (still use self.log_interval for prints)
                if self.log_interval and i_episode % self.log_interval == 0:
                    avg_len = int(avg_length / self.log_interval) if self.log_interval else 0
                    for i in range(self.n_agents):
                        if active_train[i]:
                            avg_rew = running_rewards[i] / self.log_interval
                            print(f"Episode {i_episode} | Agent {i+1}/{self.n_agents} | avg length: {avg_len} | reward: {avg_rew}")
                            with open(self._log_filename(i), 'a') as f:
                                f.write(f"{i_episode},{avg_len},{avg_rew}\n")
                            running_rewards[i] = 0.0
                    avg_length = 0

                # Dynamic early stopping check at batch boundaries (aligned with log interval)
                if dynamic and (i_episode % effective_batch == 0):
                    # Evaluate last batch average reward per agent
                    for i in range(self.n_agents):
                        if not active_train[i]:
                            continue
                        batch_avg = batch_rewards[i] / effective_batch if effective_batch else batch_rewards[i]
                        if batch_avg > TARGET_AVG:
                            consec_good_batches[i] += 1
                        else:
                            consec_good_batches[i] = 0
                        # Reset accumulator for next batch window
                        batch_rewards[i] = 0.0

                        if consec_good_batches[i] >= REQUIRED_BATCHES:
                            # Early stop this agent
                            active_train[i] = False
                            # Save at current i_episode (multiple of log interval)
                            torch.save(self.agents[i].policy.state_dict(), self._policy_filename(i, i_episode))
                            print(f"[Dynamic stop] Agent {i+1} met reward criterion. Saved policy at episode {i_episode}.")

                    # If all agents are done, end training
                    if not any(active_train):
                        break

            return

        # Sequential training: train agents one-by-one, freezing earlier ones and deploying followers progressively
        for agent_idx in range(self.n_agents):
            # Activate only followers up to current agent
            try:
                self.env.set_active_followers(agent_idx + 1)
            except Exception:
                # Backward compatibility if env doesn't have this method
                pass

            # In sequential mode, preload any existing policies for agents < current and freeze them
            if self.sequential:
                for i in range(agent_idx):
                    try:
                        basename_prev = self.get_policy(i + 1)
                        path_prev = os.path.join("Policies", basename_prev)
                        if os.path.exists(path_prev):
                            state_prev = torch.load(path_prev)
                            self.agents[i].policy.load_state_dict(state_prev)
                            self.agents[i].policy_old.load_state_dict(state_prev)
                    except Exception:
                        # Ignore if not found
                        pass

                # If a policy exists for the current agent, load and skip training it
                try:
                    basename_cur = self.get_policy(agent_idx + 1)
                    path_cur = os.path.join("Policies", basename_cur)
                    if os.path.exists(path_cur):
                        state_cur = torch.load(path_cur)
                        self.agents[agent_idx].policy.load_state_dict(state_cur)
                        self.agents[agent_idx].policy_old.load_state_dict(state_cur)
                        # Move to next agent without training
                        print(f"[Sequential] Using existing policy for Agent {agent_idx+1}: {os.path.basename(path_cur)}")
                        continue
                except Exception:
                    # Not found; proceed to training
                    pass

            # Reset per-agent accumulators
            running_rewards[agent_idx] = 0.0
            avg_length = 0
            timestep = 0
            # Ensure its memory is clear
            self.memories[agent_idx].clear_memory()

            # Dynamic early stopping trackers for this agent
            consec_good_batches = 0
            batch_reward_cur = 0.0

            # Episode counter per agent (allows unbounded if dynamic and episodes is None)
            i_episode = 0
            while True:
                i_episode += 1
                if (not dynamic) and episodes is not None and i_episode > episodes:
                    break
                states = self.env.reset(initial_condition=initial_condition, initial_condition_fn=initial_condition_fn)
                assert len(states) == self.n_agents

                # Per-episode reward accumulator for this agent
                ep_reward_cur = 0.0

                for t in range(self.max_timesteps):
                    timestep += 1
                    actions = []
                    for i in range(self.n_agents):
                        if i < agent_idx:
                            # Frozen earlier agents act greedily with their current policy
                            act = self.agents[i].policy.act_test(states[i])
                        elif i == agent_idx:
                            # Current agent collects experience
                            act = self.agents[i].policy_old.act(states[i], self.memories[i])
                        else:
                            # Not deployed yet; their actions are ignored by env
                            act = 0
                        actions.append(act)

                    next_states, rewards, done, info = self.env.step(actions)

                    # Store only current agent's reward/terminal
                    self.memories[agent_idx].rewards.append(float(rewards[agent_idx]))
                    self.memories[agent_idx].is_terminals.append(done)
                    running_rewards[agent_idx] += float(rewards[agent_idx])
                    ep_reward_cur += float(rewards[agent_idx])

                    # Periodic update for current agent only
                    if timestep % self.update_timestep == 0:
                        self.agents[agent_idx].update(self.memories[agent_idx])
                        self.memories[agent_idx].clear_memory()
                        timestep = 0

                    if self.render:
                        self.env.render()
                    if done:
                        break
                    states = next_states
                    avg_length += 1

                # After episode ends, add to batch accumulator
                batch_reward_cur += ep_reward_cur

                # Save current agent checkpoint occasionally
                if i_episode % 1000 == 0:
                    torch.save(self.agents[agent_idx].policy.state_dict(), self._policy_filename(agent_idx, i_episode))

                # Logging for current agent
                if self.log_interval and i_episode % self.log_interval == 0:
                    avg_len = int(avg_length / self.log_interval) if self.log_interval else 0
                    avg_rew = running_rewards[agent_idx] / self.log_interval
                    print(f"[Sequential] Agent {agent_idx+1}/{self.n_agents} | Episode {i_episode} | avg length: {avg_len} | reward: {avg_rew}")
                    with open(self._log_filename(agent_idx), 'a') as f:
                        f.write(f"{i_episode},{avg_len},{avg_rew}\n")
                    running_rewards[agent_idx] = 0.0
                    avg_length = 0

                # Dynamic early stopping for the current agent at batch boundaries
                if dynamic and (i_episode % effective_batch == 0):
                    batch_avg = batch_reward_cur / effective_batch if effective_batch else batch_reward_cur
                    if batch_avg > TARGET_AVG:
                        consec_good_batches += 1
                    else:
                        consec_good_batches = 0
                    batch_reward_cur = 0.0

                    if consec_good_batches >= REQUIRED_BATCHES:
                        # Save at current i_episode (multiple of log interval)
                        torch.save(self.agents[agent_idx].policy.state_dict(), self._policy_filename(agent_idx, i_episode))
                        print(f"[Sequential Dynamic stop] Agent {agent_idx+1} met reward criterion. Saved policy at episode {i_episode}.")
                        break

        return

    def create_video(self, ic=InitialCondition(distance=30, f2=1.), time=10):
        # Rollout using greedy actions from each trained agent
        state_list = self.env.reset(ic)
        frame_rate = 12
        runtime = time
        leader_positions = []
        follower_positions = [[] for _ in range(self.n_agents)]

        # Use NACA airfoil marker for swimmers
        airfoil_path = self._naca_airfoil("0017")
        fish_marker = MarkerStyle(airfoil_path, transform=mpl.transforms.Affine2D().scale(16))

        for _ in range(frame_rate * runtime):
            actions = []
            for i in range(self.n_agents):
                action = self.agents[i].policy.act_test(state_list[i])
                actions.append(action)
            state_list, _, done, info = self.env.step(actions)

            leader_positions.append((info['leader']['x'], info['leader']['y']))
            for i in range(self.n_agents):
                fx = info['followers'][i]['x']
                fy = info['followers'][i]['y']
                follower_positions[i].append((fx, fy))
            if done:
                break

        # Simple matplotlib animation (no wake dots per step to keep it light)
        fig, ax = plt.subplots(figsize=(10, 4))
        def _animate(k):
            ax.clear()
            idx_k = min(k, len(leader_positions) - 1)
            x_lead, y_lead = leader_positions[idx_k]
            ax.scatter(x_lead, y_lead, label='leader', color='black', marker=fish_marker)
            for i in range(self.n_agents):
                x_f, y_f = follower_positions[i][min(k, len(follower_positions[i]) - 1)]
                ax.scatter(x_f, y_f, label=f'follower_{i+1}', marker=fish_marker, color='black')
            # Dynamically ensure all current followers are in frame by using min current follower x
            current_x_followers = [pos[min(k, len(pos) - 1)][0] for pos in follower_positions if pos]
            x_min_all = min([x_lead] + current_x_followers) if current_x_followers else x_lead
            left_margin = 5.0
            right_margin = 5.0
            ax.set_xlim([x_min_all - left_margin, x_lead + right_margin])
            ax.set_ylim([-2. * self.env.A1, 2. * self.env.A1])
            ax.grid(True, axis='x', color='grey')

            # Draw wakes for leader and each follower with exponential decay
            T = self.env.T
            # Leader wake
            # Clip leader wake so it does not persist past follower 1 (erase-and-replace)
            if self.n_agents > 0 and follower_positions[0]:
                x_f0, _ = follower_positions[0][min(k, len(follower_positions[0]) - 1)]
            else:
                x_f0 = -numpy.inf
            for j, (xw, yw) in enumerate(leader_positions[:k]):
                if xw >= x_f0:
                    t_delay = (k - j) / frame_rate
                    wake_amp = numpy.exp(-t_delay / T)
                    ax.scatter(xw, yw * wake_amp, color='blue', s=5, marker='o')
            # Followers' wakes (each follower sheds wake for its successor)
            for i in range(self.n_agents):
                pos_list = follower_positions[i]
                for j, (xw, yw) in enumerate(pos_list[:k]):
                    # Clip follower i wake so it does not persist past follower i+1
                    if i < self.n_agents - 1 and follower_positions[i+1]:
                        x_next, _ = follower_positions[i+1][min(k, len(follower_positions[i+1]) - 1)]
                        if xw < x_next:
                            continue
                    t_delay = (k - j) / frame_rate
                    wake_amp = numpy.exp(-t_delay / T)
                    wake_noise = self.env.noise * numpy.random.normal(0, 1)
                    ax.scatter(xw, yw * wake_amp+wake_noise, color='blue', s=5, marker='o')
            return ax

        ani = animation.FuncAnimation(fig, _animate, frames=len(leader_positions), interval=frame_rate, blit=False)

        if not os.path.exists("Movies"):
            os.makedirs("Movies")

        if self.env.passive:
            ani.save(f'./Movies/swimmer_chain_N{self.n_agents}_passive.gif', writer='pillow', fps=frame_rate)
        else:
            ani.save(f'./Movies/swimmer_chain_N{self.n_agents}_active.gif', writer='pillow', fps=frame_rate)

        return

    def plot_distance(self, ic, max_timesteps=500):
        """
        Create two stacked subplots for a single initial condition:
        (1) Normalized spacing between successive swimmers (predecessor to follower),
            i.e., g_i / lambda_0, where lambda_0 = u1 * f1. Y-limit: [0, 3].
        (2) Distance from the leader to each follower X_leader - X_i (not normalized).
            Y-limit: [0, 40 * n_followers].
        """
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 7,
            'axes.titlesize': 7,
            'axes.labelsize': 7,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 7,
            'legend.title_fontsize': 7
        })

        states = self.env.reset(initial_condition=ic)
        # Per-pair spacing (predecessor to follower; length = n_followers)
        pairwise_distances = [[] for _ in range(self.n_agents)]
        # Distance from leader to each follower (length = n_followers)
        leader_distances = [[] for _ in range(self.n_agents)]
        timesteps = []

        for t in range(max_timesteps):
            actions = [self.agents[i].policy.act_test(states[i]) for i in range(self.n_agents)]
            states, _, done, info = self.env.step(actions)
            x_leader = info['leader']['x']
            for i in range(self.n_agents):
                # Pairwise distance g_i between predecessor and follower i
                pairwise_distances[i].append(info['followers'][i]['distance'])
                # Distance to leader for follower i
                leader_distances[i].append(x_leader - info['followers'][i]['x'])
            timesteps.append(t)
            if done:
                break

        # Normalization by leader wavelength (lambda_0 = u1 * f1)
        norm = ic.u1 * ic.f1

        # Build stacked subplots
        fig, (ax_norm, ax_leader) = plt.subplots(2, 1, figsize=(7, 5.0), sharex=True)

        # (1) Normalized spacing (successive swimmers)
        for i in range(self.n_agents):
            y = numpy.array(pairwise_distances[i]) / max(norm, 1e-9)
            ax_norm.plot(timesteps, y, label=f"Agent {i+1}")
        ax_norm.set_ylabel('Spacing d/$\\lambda_0$')
        ax_norm.set_ylim(0, 4)
        ax_norm.set_xlim(0, max(timesteps) if timesteps else max_timesteps)
        ax_norm.grid(False)
        ax_norm.legend(loc='upper right')

        # (2) Distance from leader to each follower
        for i in range(self.n_agents):
            ax_leader.plot(timesteps, leader_distances[i]/max(norm, 1e-9), label=f"Agent {i+1}")
        ax_leader.set_xlabel('Timesteps (0.1 s)')
        ax_leader.set_ylabel('Distance from leader (d/$\\lambda_0$)')
        # ax_leader.set_ylim(0, 1.5 * self.n_agents * ic.distance / max(norm, 1e-9))
        ax_leader.set_ylim(0, 15)
        ax_leader.set_xlim(0, max(timesteps) if timesteps else max_timesteps)
        ax_leader.grid(False)
        ax_leader.legend(loc='upper right')

        plt.tight_layout()
        return fig

    def plot_distance_all(self, initial_conditions, max_timesteps=500):
        """
        Handle a sweep of initial conditions. Plots, for each follower, the
        normalized distance trajectories for all provided initial conditions.
        """
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 7,
            'axes.titlesize': 7,
            'axes.labelsize': 7,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 7,
            'legend.title_fontsize': 7
        })

        # Create one subplot per follower
        ncols = 1
        nrows = self.n_agents
        fig, axes = plt.subplots(nrows, ncols, figsize=(7, 2.5 * self.n_agents), squeeze=False)

        for agent_idx in range(self.n_agents):
            ax = axes[agent_idx, 0]
            for ic in initial_conditions:
                # Ensure data exists or generate it
                csv_filename = self._get_data_filename_agent(ic, agent_idx)
                if not os.path.exists(csv_filename):
                    self._run_one_agent(ic, agent_idx)
                data = pd.read_csv(csv_filename)
                y = data['distance'].to_numpy() / (ic.u1 * ic.f1)
                x = data['timestep'].to_numpy()
                # Handle scalar-or-array initial conditions by selecting this agent's values
                f2_i = ic.f2[agent_idx] if hasattr(ic.f2, '__len__') else ic.f2
                A2_i = ic.A2[agent_idx] if hasattr(ic.A2, '__len__') else ic.A2
                d_i = ic.distance[agent_idx] if hasattr(ic.distance, '__len__') else ic.distance
                ax.plot(x, y, linewidth=1.5,
                        label=f"$f_{agent_idx+1}/f_{0}$ {(f2_i/ic.f1):.2f}, $A_{agent_idx+1}/A_{0}$ {(A2_i/ic.A1):.2f}, d {d_i:.1f}")

            ax.set_title(f'Agent {agent_idx+1}')
            ax.set_ylabel('Distance (d/$\\lambda_0$)')
            ax.set_ylim(0, 3.5)
            ax.set_xlim(0, max_timesteps)
            ax.grid(False)
            if agent_idx == self.n_agents - 1:
                ax.set_xlabel('Timesteps (0.1 s)')

        # Shared legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles=handles, title="Initial Conditions",
                       bbox_to_anchor=(1.02, 0.5), loc='center left')

        plt.tight_layout()
        plt.subplots_adjust(right=0.78, top=0.95, bottom=0.08)

        # Save figures
        if not os.path.exists("Figures"):
            os.makedirs("Figures")
        filename_base = f"Figures/distance_sweep_N{self.n_agents}"
        fig.savefig(f"{filename_base}.eps", format='eps', bbox_inches='tight')
        fig.savefig(f"{filename_base}.svg", format='svg', bbox_inches='tight')

        return fig

    def get_policy(self, agent_idx, episodes=None):
        """Return filename (basename only) for a specific agent's policy.
        Uses the canonical naming: ..._REW_..._noise_{noise}_{episodes}.pth
        If episodes is None, pick the latest checkpoint found in the appropriate location:
        - Sequential: Policies/ and prefix with SEQ
        - Non-sequential: project root (fallback to Policies/) and prefix with N{n_agents}
        """
        idx = self._normalize_agent_index_input(agent_idx)
        if episodes is None:
            noise_str = str(self.env.noise)
            if getattr(self, 'sequential', False):
                base_prefix = (
                    f"PPO_{self.env_name}_SEQ_AG{idx+1}_OBS_{self.env.observations}_REW_{self.env.rewards}_"
                )
                search_dirs = ["Policies"]
            else:
                base_prefix = (
                    f"PPO_{self.env_name}_N{self.n_agents}_AG{idx+1}_OBS_{self.env.observations}_REW_{self.env.rewards}_"
                )
                search_dirs = [".", "Policies"]
            # Canonical order: noise first, then episode
            pattern_noise_then_ep = re.compile(
                re.escape(base_prefix) + r"noise_" + re.escape(noise_str) + r"_(\d+)\.pth$"
            )
            best = None  # tuple of (episode, full_path)
            for d in search_dirs:
                if not os.path.isdir(d):
                    continue
                for f in os.listdir(d):
                    if getattr(self, 'sequential', False):
                        if not f.startswith(f"PPO_{self.env_name}_SEQ_AG{idx+1}_"):
                            continue
                    else:
                        if not f.startswith(f"PPO_{self.env_name}_N{self.n_agents}_AG{idx+1}_"):
                            continue
                    if not f.endswith('.pth'):
                        continue
                    if not f.startswith(base_prefix):
                        continue
                    ep = None
                    m = pattern_noise_then_ep.search(f)
                    if m:
                        ep = int(m.group(1))
                    if ep is not None:
                        full_path = os.path.join(d, f)
                        if (best is None) or (ep > best[0]):
                            best = (ep, full_path)
                if best is not None:
                    break
            if best is None:
                raise FileNotFoundError(f"No saved policies found for agent {agent_idx}")
            return os.path.basename(best[1])
        # If a specific episode is requested, return the canonical basename (noise before episodes)
        if getattr(self, 'sequential', False):
            return (
                f"PPO_{self.env_name}_SEQ_AG{idx+1}_OBS_{self.env.observations}_REW_{self.env.rewards}_noise_{self.env.noise}_{episodes}.pth"
            )
        return f"PPO_{self.env_name}_N{self.n_agents}_AG{idx+1}_OBS_{self.env.observations}_REW_{self.env.rewards}_noise_{self.env.noise}_{episodes}.pth"

    def visualize_policy_agent(self, agent_idx, resolutions=[15, 15, 5], ic=None):
        """
        Visualize policy map for a specific agent over [f2, A2, |v_wake|].
        Requires env.observations to include exactly ['f2','A2','v_flow'] for meaningful axes.
        """
        idx = self._normalize_agent_index_input(agent_idx)

        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 7,
            'axes.titlesize': 7,
            'axes.labelsize': 7,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 7,
            'legend.title_fontsize': 7
        })

        # Axes ranges analogous to single-agent
        obs_low = numpy.array([0, 0, 0])
        obs_high = numpy.array([2, 4., 0.75])

        trajectory = None
        v_flow_values = []

        if ic is not None:
            # Run a short rollout to collect this agent's trajectory in [f2, A2, |v_wake|]
            states = self.env.reset(initial_condition=ic)
            for _ in range(50):
                actions = [self.agents[i].policy.act_test(states[i]) for i in range(self.n_agents)]
                states, _, done, info = self.env.step(actions)
                # Observation components for this agent
                f2 = self.env.f_f[idx]
                A2 = self.env.A_f[idx]
                vabs = abs(info['followers'][idx]['v_flow'])
                v_flow_values.append(vabs)
                pt = numpy.array([f2, A2, vabs], dtype=numpy.float32)
                trajectory = pt if trajectory is None else numpy.vstack([trajectory, pt])
                if done:
                    break

            if v_flow_values:
                obs_high[2] = max(max(v_flow_values), obs_high[2])

        # Build grid and evaluate this agent's policy
        grids = [numpy.linspace(low, high, res) for low, high, res in zip(obs_low, obs_high, resolutions)]
        X, Y, Z = numpy.meshgrid(*grids, indexing='ij')
        pts = numpy.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        actions = numpy.array([self.agents[idx].policy.act_test(pt) for pt in pts])

        fig = plt.figure(figsize=(6, 6.5))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=actions, cmap='tab10', alpha=0.6 if trajectory is None else 0.3, s=50)

        if trajectory is not None:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='orange', linewidth=3, marker='o', markersize=4, label='Trajectory', alpha=0.9)
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='green', s=100, marker='s', label='Start', alpha=1.0)

        ax.set_xlabel(f'$f_{{{idx+1}}}$')
        ax.set_ylabel(f'$A_{{{idx+1}}}$')
        ax.set_zlabel('$|v_{wake}|$')
        zmin, zmax = obs_low[2], obs_high[2]
        ax.set_zlim(zmin, zmax)
        ax.set_box_aspect([1, 1, 1])

        action_labels = [
            f'Decrease $f_{{{idx+1}}}$', f'Increase $f_{{{idx+1}}}$',
            f'Decrease $A_{{{idx+1}}}$', f'Increase $A_{{{idx+1}}}$'
        ]
        handles = []
        unique_actions = numpy.unique(actions)
        for a in unique_actions:
            handles.append(mpatches.Patch(color=sc.cmap(sc.norm(a)), label=action_labels[int(a)]))
        ax.legend(handles=handles, title=f"Agent {idx+1} Action", loc='lower left', bbox_to_anchor=(1.1, 0.5))

        plt.tight_layout()

        # Save figure
        if not os.path.exists("Figures"):
            os.makedirs("Figures")
        if ic is None:
            filename_base = f"Figures/policy_map_agent_{idx+1}_N{self.n_agents}"
        else:
            if not os.path.exists("Figures/Policy Paths"):
                os.makedirs("Figures/Policy Paths")
            f2_i = ic.f2[idx] if hasattr(ic.f2, '__len__') else ic.f2
            A2_i = ic.A2[idx] if hasattr(ic.A2, '__len__') else ic.A2
            d_i = ic.distance[idx] if hasattr(ic.distance, '__len__') else ic.distance
            ic_str = f"d_{d_i:.1f}_f{idx+1}_{f2_i:.1f}_A{idx+1}_{A2_i:.1f}"
            filename_base = f"Figures/Policy Paths/policy_path_agent_{idx+1}_N{self.n_agents}_{ic_str}"
        fig.savefig(f"{filename_base}.svg", format='svg')
        fig.savefig(f"{filename_base}.eps", format='eps')

        return fig

    def _get_data_filename_agent(self, ic, agent_idx):
        tag = 'passive' if self.env.passive else 'active'
        # Use 1-based AG id in filename
        f2_i = ic.f2[agent_idx] if hasattr(ic.f2, '__len__') else ic.f2
        A2_i = ic.A2[agent_idx] if hasattr(ic.A2, '__len__') else ic.A2
        d_i = ic.distance[agent_idx] if hasattr(ic.distance, '__len__') else ic.distance
        return f"All data/time_distance_N{self.n_agents}_AG_{agent_idx+1}_f{agent_idx+1}_{f2_i:.2f}_A{agent_idx+1}_{A2_i:.2f}_d_{d_i:.1f}_{tag}.csv"

    def _run_one_agent(self, ic, agent_idx):
        distances = []
        times = []
        states = self.env.reset(initial_condition=ic)
        # clear memories irrelevant here
        for t in range(500):
            actions = [self.agents[i].policy.act_test(states[i]) for i in range(self.n_agents)]
            states, _, done, info = self.env.step(actions)
            distances.append(info['followers'][agent_idx]['distance'])
            times.append(t)
            if done:
                break
        if not os.path.exists("All data"):
            os.makedirs("All data")
        csv_filename = self._get_data_filename_agent(ic, agent_idx)
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestep', 'distance'])
                for timestep, dist in zip(times, distances):
                    writer.writerow([timestep, dist])
        else:
            print(f"File {csv_filename} already exists, skipping...")

    def plot_distance_agent(self, initial_conditions, agent_idx, max_timesteps=500):
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 7,
            'axes.titlesize': 7,
            'axes.labelsize': 7,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 7,
            'legend.title_fontsize': 7
        })

        fig, ax = plt.subplots(figsize=(7, 2.5))
        idx = self._normalize_agent_index_input(agent_idx)
        for ic in initial_conditions:
            old_passive = self.env.passive
            csv_filename = self._get_data_filename_agent(ic, idx)
            if os.path.exists(csv_filename):
                data = pd.read_csv(csv_filename)
            else:
                # create if missing
                self._run_one_agent(ic, idx)
                data = pd.read_csv(csv_filename)
            # Normalize distance by leader wavelength like single-agent
            data['schooling_number'] = data['distance'] / (ic.u1 * ic.f1)
            f2_i = ic.f2[idx] if hasattr(ic.f2, '__len__') else ic.f2
            A2_i = ic.A2[idx] if hasattr(ic.A2, '__len__') else ic.A2
            ax.plot(
                data['timestep'], data['schooling_number'],
                label=f"Agent {idx+1}: $f_{idx+1}/f_{0}$ {(f2_i/ic.f1):.2f}, $A_{idx+1}/A_{0}$ {(A2_i/ic.A1):.2f}"
            )
            self.env.passive = old_passive

        ax.set_xlabel('Timesteps (0.1 s)')
        ax.set_ylabel('Distance (d/$\\lambda_0$)')
        ax.set_ylim(0, 3)
        ax.set_xlim(0, max_timesteps)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1, title="Initial Conditions")
        ax.grid(False)
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.7, top=0.9, bottom=0.15)
        return fig

    def test_all_agent(self, agent_idx, n=25, legend=True, save_figures=True):
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 8,
            'axes.titlesize': 8,
            'axes.labelsize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 8,
            'legend.title_fontsize': 8
        })
        f_values = numpy.linspace(0.01, 2., n)
        a_values = numpy.linspace(0.01, 4., n)
        d_values = numpy.linspace(15., 60., 10)
        ff, aa, dd = numpy.meshgrid(f_values, a_values, d_values)
        initial_conditions = [InitialCondition(distance=dd, f2=ff, A2=aa) for ff, aa, dd in zip(ff.flatten(), aa.flatten(), dd.flatten())]

        colors = ['#440154', '#21908C', '#FDE725']
        labels = ['Stable', 'Collide', 'Separate']

        fig, (ax_passive, ax_active) = plt.subplots(1, 2, figsize=(8, 3.5))
        passive_conditions = [True, False]
        axes = [ax_passive, ax_active]
        titles = ['Passive', 'Active']

        for passive, ax, title in zip(passive_conditions, axes, titles):
            rows = []
            old_passive = self.env.passive
            self.env.passive = passive
            idx = self._normalize_agent_index_input(agent_idx)
            for ic in initial_conditions:
                csv_filename = self._get_data_filename_agent(ic, idx)
                if not os.path.exists(csv_filename):
                    self._run_one_agent(ic, idx)
                data = pd.read_csv(csv_filename)
                if data['distance'].iloc[-1] < 0.5:
                    result = 1
                elif data['distance'].iloc[-1] > 75:
                    result = 2
                else:
                    result = 0
                f2_i = ic.f2[idx] if hasattr(ic.f2, '__len__') else ic.f2
                A2_i = ic.A2[idx] if hasattr(ic.A2, '__len__') else ic.A2
                d_i = ic.distance[idx] if hasattr(ic.distance, '__len__') else ic.distance
                rows.append({'f_i/f_0': f2_i/ic.f1, 'A_i/A_0': A2_i/ic.A1, 'd': d_i, 'Result': result})
            self.env.passive = old_passive

            results = pd.DataFrame(rows, columns=['f_i/f_0', 'A_i/A_0', 'd', 'Result'])
            grouped = results.groupby(['f_i/f_0', 'A_i/A_0'])['Result'].value_counts().unstack(fill_value=0)
            for col in [0, 1, 2]:
                if col not in grouped.columns:
                    grouped[col] = 0
            grouped = grouped[[0, 1, 2]]

            pie_size = 0.03
            for (f2_ratio, A2_ratio), counts in grouped.iterrows():
                total = counts.sum()
                if total == 0:
                    continue
                sizes = counts.values / total
                wedge_colors = []
                wedge_sizes = []
                for i, size in enumerate(sizes):
                    if size > 0:
                        wedge_colors.append(colors[i])
                        wedge_sizes.append(size)
                if not wedge_sizes:
                    continue
                x_pos = f2_ratio
                y_pos = A2_ratio
                wedges, _ = ax.pie(wedge_sizes, colors=wedge_colors, center=(x_pos, y_pos), radius=pie_size, startangle=90)
                for wedge in wedges:
                    wedge.set_edgecolor('black')
                    wedge.set_linewidth(0.5)

            ax.set_xlabel('$f_i/f_0$')
            ax.set_ylabel('$A_i/A_0$')
            ax.set_title(f'{title} (Agent {idx+1})', pad=20)
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            ax.set_xticks(numpy.arange(0, 2.5, 0.5))
            ax.set_yticks(numpy.arange(0, 2.5, 0.5))
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

        if legend:
            legend_elements = [
                mpatches.Patch(color=colors[0], label=labels[0]),
                mpatches.Patch(color=colors[1], label=labels[1]),
                mpatches.Patch(color=colors[2], label=labels[2]),
            ]
            fig.legend(handles=legend_elements, title="Outcome", bbox_to_anchor=(0.9, 0.5), loc="center left")
        plt.tight_layout()
        if legend:
            fig.subplots_adjust(right=0.88, top=0.9)
        if not os.path.exists("Figures"):
            os.makedirs("Figures")
        filename_base = f"Figures/stability_test_agent_{idx+1}_N{self.n_agents}"
        if save_figures:
            fig.savefig(f"{filename_base}.eps", format='eps', bbox_inches='tight')
            fig.savefig(f"{filename_base}.svg", format='svg', bbox_inches='tight')
        return fig

    def plot_distance_all_agent(self, agent_idx, max_timesteps=200):
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 7,
            'axes.titlesize': 7,
            'axes.labelsize': 7,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6,
            'legend.fontsize': 7,
            'legend.title_fontsize': 7
        })
        freq = numpy.linspace(0.5, 1.5, 11)
        amps = numpy.linspace(1., 4., 11)
        dists = numpy.linspace(15, 60, 11)
        fixed_distance = 20.0
        fixed_f2 = 1.0
        fixed_A2 = 2.0

        passive_conditions = [True, False]
        condition_names = ['passive', 'active']

        for passive, condition_name in zip(passive_conditions, condition_names):
            fig, axes = plt.subplots(3, 1, figsize=(6, 5))
            cmap = plt.cm.gray_r
            idx = self._normalize_agent_index_input(agent_idx)

            # 1) varying distance
            ax = axes[0]
            colors = cmap(numpy.linspace(0.2, 0.8, len(dists)))
            ic_ref = InitialCondition(distance=dists[0], f2=fixed_f2, A2=fixed_A2)
            normalized_dists = dists / (ic_ref.u1 * ic_ref.f1)
            for i, d in enumerate(dists):
                ic = InitialCondition(distance=d, f2=fixed_f2, A2=fixed_A2)
                csv_filename = self._get_data_filename_agent(ic, idx)
                if os.path.exists(csv_filename):
                    data = pd.read_csv(csv_filename)
                else:
                    self._run_one_agent(ic, idx)
                    data = pd.read_csv(csv_filename)
                data['schooling_number'] = data['distance'] / (ic.u1 * ic.f1)
                ax.plot(data['timestep'], data['schooling_number'], color=colors[i], linewidth=2)
            ax.set_ylabel('Distance (d/$\\lambda_1$)')
            ax.set_ylim(0, 3.5)
            ax.set_xlim(0, max_timesteps)
            ax.grid(False)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=normalized_dists.min(), vmax=normalized_dists.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('Distance (d/$\\lambda_0$)')

            # 2) varying f2
            ax = axes[1]
            colors = cmap(numpy.linspace(0.2, 0.8, len(freq)))
            ic_ref = InitialCondition(distance=fixed_distance, f2=freq[0], A2=fixed_A2)
            freq_ratios = freq / ic_ref.f1
            for i, f in enumerate(freq):
                ic = InitialCondition(distance=fixed_distance, f2=f, A2=fixed_A2)
                csv_filename = self._get_data_filename_agent(ic, idx)
                if os.path.exists(csv_filename):
                    data = pd.read_csv(csv_filename)
                else:
                    self._run_one_agent(ic, idx)
                    data = pd.read_csv(csv_filename)
                data['schooling_number'] = data['distance'] / (ic.u1 * ic.f1)
                ax.plot(data['timestep'], data['schooling_number'], color=colors[i], linewidth=2)
            ax.set_ylabel('Distance (d/$\\lambda_0$)')
            ax.set_ylim(0, 3.5)
            ax.set_xlim(0, max_timesteps)
            ax.grid(False)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=freq_ratios.min(), vmax=freq_ratios.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('$f_i/f_0$')

            # 3) varying A2
            ax = axes[2]
            colors = cmap(numpy.linspace(0.2, 0.8, len(amps)))
            ic_ref = InitialCondition(distance=fixed_distance, f2=fixed_f2, A2=amps[0])
            amp_ratios = amps / ic_ref.A1
            for i, a in enumerate(amps):
                ic = InitialCondition(distance=fixed_distance, f2=fixed_f2, A2=a)
                csv_filename = self._get_data_filename_agent(ic, idx)
                if os.path.exists(csv_filename):
                    data = pd.read_csv(csv_filename)
                else:
                    self._run_one_agent(ic, idx)
                    data = pd.read_csv(csv_filename)
                data['schooling_number'] = data['distance'] / (ic.u1 * ic.f1)
                ax.plot(data['timestep'], data['schooling_number'], color=colors[i], linewidth=2)
            ax.set_xlabel('Timesteps (0.1 s)')
            ax.set_ylabel('Distance (d/$\\lambda_0$)')
            ax.set_ylim(0, 3.5)
            ax.set_xlim(0, max_timesteps)
            ax.grid(False)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=amp_ratios.min(), vmax=amp_ratios.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('$A_i/A_0$')

            axes[0].set_xticklabels([])
            axes[1].set_xticklabels([])
            plt.tight_layout()

            if not os.path.exists("Figures"):
                os.makedirs("Figures")
            filename_base = f"Figures/distance_all_agent_{idx+1}_N{self.n_agents}_{condition_name}"
            fig.savefig(f"{filename_base}.eps", format='eps', bbox_inches='tight')
            fig.savefig(f"{filename_base}.svg", format='svg', bbox_inches='tight')

        return


    def _naca_airfoil(self, code, num_points=100):
        """Generates the coordinates of a NACA 4-digit airfoil."""
        m = float(code[0]) / 100.0  # Maximum camber
        p = float(code[1]) / 10.0  # Location of maximum camber
        t = float(code[2:]) / 100.0  # Maximum thickness

        x = numpy.linspace(0, 1, num_points)
        yt = 5 * t * (0.2969 * numpy.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

        if m == 0 and p == 0:
            yc = numpy.zeros_like(x)
            theta = numpy.zeros_like(x)
        else:
            yc = numpy.where(x <= p,
                            m / p**2 * (2 * p * x - x**2),
                            m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
            theta = numpy.arctan(numpy.gradient(yc, x))

        xu = x - yt * numpy.sin(theta)
        yu = yc + yt * numpy.cos(theta)
        xl = x + yt * numpy.sin(theta)
        yl = yc - yt * numpy.cos(theta)

        # Close the path by adding the first point to the end
        x_coords = -numpy.concatenate([xu, xl[::-1]])
        y_coords = numpy.concatenate([yu, yl[::-1]])

        return mpath.Path(numpy.column_stack([x_coords, y_coords]))