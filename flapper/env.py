import random
import numpy
import scipy
import matplotlib.pyplot as plt
plt.ioff() 
import gym
from gym import wrappers
from gym import spaces

import imageio

# Module-level default physical constants (decouple from single-agent env)
DEFAULT_S = 15.0
DEFAULT_C = 4.0
DEFAULT_AS = DEFAULT_S * DEFAULT_C
DEFAULT_T = 1.0
DEFAULT_M = 80.0
DEFAULT_CT = 0.96
DEFAULT_CD = 0.25
DEFAULT_RHO = 1.0
DEFAULT_DT = 0.1


class InitialCondition(object):

    def __init__(self, distance=None, f2=None, A2=None, f1=None, A1=None, goal=None):
        # Leader is indexed as 0 (A0, f0). Keep backward-compatible kwargs f1/A1 (leader) and f2/A2 (follower default).
        self.distance = distance if distance is not None else 21.5
        # Rename conceptually: leader -> index 0
        self.A1 = 2.0 if A1 is None else A1
        self.f1 = 1.0 if f1 is None else f1
        # Single follower defaults retained for backward compatibility
        self.A2 = 2.0 if A2 is None else A2
        self.f2 = 1.0 if f2 is None else f2
        self.goal = goal if goal is not None else 21.5
        self.u1 = numpy.pi * self.A1 * self.f1 * numpy.sqrt(2 * DEFAULT_CT / DEFAULT_CD)

        self.u2 = numpy.pi * self.A2 * self.f2 * numpy.sqrt(2 * DEFAULT_CT / DEFAULT_CD)
        self.v2 = -self.A2*(2 * numpy.pi * self.f2)
        self.t_delay = -self.distance/self.u2
        # Prevent overflow in exponential by clipping the argument
        exp_arg = numpy.clip(-self.t_delay/DEFAULT_T, -700, 700)
        self.v_flow = self.A1*self.f1*numpy.cos(2*numpy.pi*self.f1*(-self.t_delay))*numpy.exp(exp_arg)
        self.flow_agreement = self.v2 * self.v_flow # Flow agreement

    def random(self, randomize_fields=[]):
        if 'distance' in randomize_fields:
            self.distance = random.uniform(10, 60)
        if 'f2' in randomize_fields:
            self.f2 = random.uniform(0.5, 1.5)
        if 'A2' in randomize_fields:
            self.A2 = random.uniform(1., 4.0)
        if 'v2' in randomize_fields:
            self.v2 = random.uniform(-1.0, 1.0)
        if 'A1' in randomize_fields:
            self.A1 = random.uniform(1., 4.0)
        if 'f1' in randomize_fields:
            self.f1 = random.uniform(0.5, 1.5)
        return self

class MultiSwimmerEnv(gym.Env):
    """
    Chain of swimmers: 1 leader (index -1, implicit) + N followers.
    Erase-and-replace wake: follower i only interacts with the wake from swimmer i-1.

    Observations/rewards are defined per follower and returned as lists in the same order.
    Default behavior with n_followers=1 matches the single-follower setup conceptually.
    """

    # Physical constants (copied from single-agent defaults; independent here)
    s = DEFAULT_S
    c = DEFAULT_C
    As = DEFAULT_AS
    T = DEFAULT_T
    m = DEFAULT_M
    Ct = DEFAULT_CT
    Cd = DEFAULT_CD
    rho = DEFAULT_RHO
    dt = DEFAULT_DT

    def __init__(self, n_followers=1, observations=['f2','A2','v_flow'], rewards=['distance'], history_length=1, passive=False, noise = 0.0):
        super(MultiSwimmerEnv, self).__init__()

        if n_followers < 1:
            raise ValueError("n_followers must be >= 1")

        self.n_followers = int(n_followers)
        self.n_history = history_length
        self.observations = observations
        self.rewards = rewards
        self.passive = passive
        self.noise = noise
        # Number of followers currently deployed/active in the simulation.
        # In sequential training we can increase this from 1 up to n_followers.
        self.active_followers_count = self.n_followers

        # Each follower chooses among 4 discrete actions
        self.action_space = spaces.Tuple([spaces.Discrete(4) for _ in range(self.n_followers)])

        # Observation space per agent: same dimensionality as single-agent, repeated for history
        n_obs = len(self.observations)
        obs_low = [-numpy.inf for _ in range(n_obs*self.n_history)]
        obs_high = [numpy.inf for _ in range(n_obs*self.n_history)]
        self.observation_space_per_agent = spaces.Box(low=numpy.array(obs_low), high=numpy.array(obs_high), dtype=numpy.float32)

        # Aggregate space for vectorized compatibility (not used by trainer directly)
        self.observation_space = spaces.Tuple([self.observation_space_per_agent for _ in range(self.n_followers)])

        self.t_bound = 500.

        # State containers initialized in reset()
        self.reset()

    def set_active_followers(self, k):
        """Set how many followers are active (1..n_followers).
        Followers with index >= k are ignored during dynamics, rewards, and termination.
        """
        if not (1 <= int(k) <= self.n_followers):
            raise ValueError(f"active followers count must be in 1..{self.n_followers}, got {k}")
        self.active_followers_count = int(k)

    def _shoot_pair(self, A_prev, A_cur, f_prev, f_cur, vec_initial, x_prev_start, u_advect, t_start=0., t_bound=5000., method='RK45'):
        rho = self.rho
        As = self.As
        T = self.T
        m = self.m
        Ct = self.Ct
        Cd = self.Cd

        def fun(t, vec):
            x_cur, u_cur = vec
            # Linearized predecessor x-position during [t_start, t_bound]
            x_prev_t = x_prev_start + u_advect * (t - t_start)
            t_delay = (x_prev_t - x_cur) / max(u_advect, 1e-6)
            noise = self.noise * numpy.random.normal(0, 1)
            Ft = 2*rho*As*Ct*numpy.pi**2*(A_cur*f_cur*numpy.cos(2*numpy.pi*f_cur*t)
                                          - A_prev*f_prev*numpy.cos(2*numpy.pi*f_prev*(t - t_delay))*numpy.exp(-t_delay/T)+noise)**2
            Fd = rho*As*Cd*u_cur**2/2
            dy_dt = (u_cur, (Ft - Fd)/m)
            return numpy.asarray(dy_dt)

        solver = scipy.integrate.solve_ivp(fun, (t_start, t_bound), vec_initial, method=method,
                                            rtol=1e-4, atol=1e-7, max_step=.03, first_step=.001, dense_output=True)
        values = list(zip(solver.t, solver.y.T))
        info = {
            'x': values[-1][1][0],
            'u': values[-1][1][1],
        }
        return solver, values, info

    def _get_obs_for_agent(self, i):
        obs = []
        if 'f2' in self.observations:
            obs.append(self.f_f[i])
        if 'A2' in self.observations:
            obs.append(self.A_f[i])
        if 'distance' in self.observations:
            obs.append(self.distance[i])
        if 'v2' in self.observations:
            obs.append(self.v_f[i])
        if 'v_flow' in self.observations:
            obs.append(self.avg_v_flow[i])
        if 'u2' in self.observations:
            obs.append(self.avg_u[i])
        return numpy.array(obs, dtype=numpy.float32)

    def step(self, actions):
        # Normalize actions list/tuple
        if isinstance(actions, (list, tuple)):
            actions_seq = list(actions)
        else:
            # Single action broadcast if n_followers==1
            actions_seq = [actions]

        if len(actions_seq) != self.n_followers:
            raise ValueError(f"Expected {self.n_followers} actions, got {len(actions_seq)}")

        # Apply actions to follower parameters if active
        if not self.passive:
            for i, action in enumerate(actions_seq):
                if action == 0:
                    self.f_f[i] = max(self.f_f[i] - 0.1, 0.01)
                elif action == 1:
                    self.f_f[i] = min(self.f_f[i] + 0.1, 2.0)
                elif action == 2:
                    self.A_f[i] = max(self.A_f[i] - 0.1, 0.01)
                elif action == 3:
                    self.A_f[i] = min(self.A_f[i] + 0.1, 4.0)

        # Advance dynamics over one small step for all followers, using erase-and-replace wake
        t_bound_step = self.dt
        prev_flap = self.flap.copy()
        prev_u = self.u_prev.copy()

        # Leader instantaneous state at t_start
        self.u1 = numpy.pi * self.A1 * self.f1 * numpy.sqrt(2 * self.Ct / self.Cd)
        x1_start = self.u1 * self.tt
        y1 = self.A1 * numpy.sin(2 * numpy.pi * self.f1 * self.tt)

        # Integrate each follower sequentially using predecessor state frozen at t_start
        new_flap = numpy.zeros_like(self.flap)
        shoot_infos = []
        active_n = int(self.active_followers_count)
        for i in range(active_n):
            if i == 0:
                A_prev, f_prev = self.A1, self.f1
                x_prev_start = x1_start
                u_adv = self.u1
            else:
                A_prev, f_prev = self.A_f[i-1], self.f_f[i-1]
                x_prev_start = prev_flap[i-1, 0]
                u_adv = max(prev_flap[i-1, 1], 1e-6)

            solver, values, info = self._shoot_pair(
                A_prev, self.A_f[i], f_prev, self.f_f[i],
                self.flap[i], x_prev_start, u_adv,
                t_start=self.tt, t_bound=self.tt + t_bound_step
            )
            new_flap[i] = values[-1][1]
            shoot_infos.append(info)

        # Inactive followers keep previous state (do not advance) to avoid interfering
        for i in range(active_n, self.n_followers):
            new_flap[i] = prev_flap[i]

        # Commit state update
        self.flap = new_flap
        self.tt += self.dt

        # Compute observations, rewards, and info per follower
        rewards = [0.0 for _ in range(self.n_followers)]
        dones = [False for _ in range(self.n_followers)]

        # Derive v_flow and other metrics with erase-and-replace for each follower
        self.avg_u = numpy.zeros(self.n_followers)
        self.v_f = numpy.zeros(self.n_followers)
        self.distance = numpy.zeros(self.n_followers)
        v_flow_list = []
        v_flow_head_list = []
        v_gradient_list = []
        a_flow_list = []

        for i in range(self.n_followers):
            # Skip inactive followers entirely for metrics/rewards/termination
            if i >= self.active_followers_count:
                # Maintain histories minimally for shape consistency
                self.u_histories[i].append(self.flap[i, 1])
                if len(self.u_histories[i]) > 10:
                    self.u_histories[i].pop(0)
                self.avg_u[i] = numpy.mean(self.u_histories[i]) if self.u_histories[i] else 0.0
                self.v_flow_histories[i].append(0.0)
                if len(self.v_flow_histories[i]) > 1:
                    self.v_flow_histories[i].pop(0)
                self.avg_v_flow[i] = numpy.mean(self.v_flow_histories[i]) if self.v_flow_histories[i] else 0.0
                self.distance[i] = self.distance[i]
                continue
            # Current follower states
            x_i = self.flap[i, 0]
            u_i = self.flap[i, 1]

            # Predecessor at t = current time (we use updated positions for reporting)
            if i == 0:
                x_prev = self.u1 * self.tt
                A_prev, f_prev = self.A1, self.f1
                u_adv = self.u1
            else:
                x_prev = self.flap[i-1, 0]
                A_prev, f_prev = self.A_f[i-1], self.f_f[i-1]
                u_adv = max(self.flap[i-1, 1], 1e-6)

            # Wake at tail and head of follower i due to predecessor only
            t_delay = (x_prev - x_i) / max(u_adv, 1e-6)
            # Prevent overflow in exponential by clipping the argument
            exp_arg_flow = numpy.clip(-t_delay/self.T, -700, 700)
            v_flow = A_prev*f_prev*numpy.cos(2*numpy.pi*f_prev*(self.tt - t_delay))*numpy.exp(exp_arg_flow)

            t_delay_head = (x_prev - (x_i + self.c)) / max(u_adv, 1e-6)
            exp_arg_head = numpy.clip(-t_delay_head/self.T, -700, 700)
            v_flow_head = A_prev*f_prev*numpy.cos(2*numpy.pi*f_prev*(self.tt - t_delay_head))*numpy.exp(exp_arg_head)

            v_gradient = (v_flow_head - v_flow)/self.c

            # Stats histories per agent
            self.u_histories[i].append(u_i)
            if len(self.u_histories[i]) > 10:
                self.u_histories[i].pop(0)
            self.avg_u[i] = numpy.mean(self.u_histories[i])

            self.v_flow_histories[i].append(abs(v_flow))
            if len(self.v_flow_histories[i]) > 1:
                self.v_flow_histories[i].pop(0)
            self.avg_v_flow[i] = numpy.mean(self.v_flow_histories[i])

            # Flow agreement metric
            v_i = -self.A_f[i]*(2 * numpy.pi * self.f_f[i])*numpy.cos(2 * numpy.pi * self.f_f[i] * self.tt)
            flow_agreement = v_i * v_flow_head
            self.flow_agreement_histories[i].append(flow_agreement)
            if len(self.flow_agreement_histories[i]) > 10:
                self.flow_agreement_histories[i].pop(0)
            avg_flow_agreement = numpy.mean(self.flow_agreement_histories[i])

            # Distance to predecessor (positive if behind predecessor)
            dist_i = x_prev - x_i - self.c
            self.distance[i] = dist_i
            self.distance_histories[i].append(dist_i)
            if len(self.distance_histories[i]) > 500:
                self.distance_histories[i].pop(0)
            avg_distance = numpy.mean(self.distance_histories[i])
            distance_from_average = abs(dist_i - avg_distance)

            # Reward shaping terms similar to single-agent
            reward_i = 0.0
            if 'distance' in self.rewards:
                reward_i += 1.0 * (self.prev_distance_from_average[i] - distance_from_average)
            if 'flow_agreement' in self.rewards:
                reward_i += 1.0 * (avg_flow_agreement - self.prev_avg_flow_agreement[i])

            # Penalties/termination for extreme behaviors
            done_i = False
            if dist_i < 0.0 or dist_i > 200.0:
                reward_i -= 100.0
                done_i = True

            # Save running stats
            rewards[i] = reward_i
            dones[i] = done_i
            self.prev_avg_flow_agreement[i] = avg_flow_agreement
            self.prev_distance_from_average[i] = distance_from_average

            # Lists for info
            v_flow_list.append(v_flow)
            v_flow_head_list.append(v_flow_head)
            v_gradient_list.append(v_gradient)
            a_flow_list.append((v_flow - self.prev_v_flow[i]) / self.dt)
            self.prev_v_flow[i] = v_flow

        # Build observation histories per follower
        obs_list = []
        for i in range(self.n_followers):
            self.obs_histories[i].append(self._get_obs_for_agent(i))
            if len(self.obs_histories[i]) > self.n_history:
                self.obs_histories[i].pop(0)
            obs_list.append(numpy.array(self.obs_histories[i]).flatten())

        # Compose info dict
        info = {
            't': self.tt,
            'leader': {'x': self.u1 * self.tt, 'y': y1, 'u': self.u1, 'A1': self.A1, 'f1': self.f1},
            'followers': [
                {
                    'x': float(self.flap[i, 0]),
                    'y': float(self.A_f[i] * numpy.sin(2 * numpy.pi * self.f_f[i] * self.tt)),
                    'u': float(self.flap[i, 1]),
                    'f': float(self.f_f[i]),
                    'A': float(self.A_f[i]),
                    'distance': float(self.distance[i]),
                    'v_flow': float(v_flow_list[i]) if i < len(v_flow_list) else 0.0,
                    'v_flow_head': float(v_flow_head_list[i]) if i < len(v_flow_head_list) else 0.0,
                    'avg_v_flow': float(self.avg_v_flow[i]) if i < len(self.avg_v_flow) else 0.0,
                    'v_gradient': float(v_gradient_list[i]) if i < len(v_gradient_list) else 0.0,
                }
                for i in range(self.n_followers)
            ]
        }

        # Global done: any active follower done
        done_global = any(dones[:self.active_followers_count])
        return obs_list, rewards, done_global, info

    def reset(self, initial_condition=None, initial_condition_fn=None):
        # Base initial condition
        if initial_condition is None:
            if initial_condition_fn is not None:
                initial_condition = initial_condition_fn()
            else:
                initial_condition = InitialCondition()

        # Leader parameters
        self.A1 = initial_condition.A1
        self.f1 = initial_condition.f1

        # Followers initial parameters (all equal by default)
        self.A_f = numpy.array([initial_condition.A2 for _ in range(self.n_followers)], dtype=float)
        self.f_f = numpy.array([initial_condition.f2 for _ in range(self.n_followers)], dtype=float)
        self.goal = initial_condition.goal

        # Initialize follower states along x behind leader with equal spacing = distance
        base_distance = float(initial_condition.distance)
        u_initial = float(initial_condition.u2)

        self.flap = numpy.zeros((self.n_followers, 2), dtype=float)
        for i in range(self.n_followers):
            # Follower i starts at - (i+1) * distance
            self.flap[i] = numpy.asarray([-(i+1) * base_distance, u_initial])

        self.tt = 0.0

        # Histories per follower
        self.flow_agreement_histories = [[] for _ in range(self.n_followers)]
        self.u_histories = [[] for _ in range(self.n_followers)]
        self.v_flow_histories = [[] for _ in range(self.n_followers)]
        self.distance_histories = [[(i+1)*base_distance for _ in range(499)] for i in range(self.n_followers)]
        self.prev_v_flow = [0.0 for _ in range(self.n_followers)]
        self.prev_avg_flow_agreement = [0.0 for _ in range(self.n_followers)]
        self.prev_distance_from_average = [0.0 for _ in range(self.n_followers)]
        self.u_prev = self.flap[:, 1].copy()

        # Averages placeholders
        self.avg_v_flow = numpy.zeros(self.n_followers)
        self.avg_u = numpy.zeros(self.n_followers)
        self.v_f = numpy.zeros(self.n_followers)
        self.distance = numpy.array([(i+1)*base_distance for i in range(self.n_followers)], dtype=float)

        # Observation histories
        self.obs_histories = [[] for _ in range(self.n_followers)]
        for i in range(self.n_followers):
            self.obs_histories[i] = [self._get_obs_for_agent(i) for _ in range(self.n_history)]

        obs_list = [numpy.array(self.obs_histories[i]).flatten() for i in range(self.n_followers)]
        return obs_list

    def render(self, mode='rgb_array'):
        fig, ax = plt.subplots()
        x1 = self.u1 * self.tt
        y1 = self.A1 * numpy.sin(2 * numpy.pi * self.f1 * self.tt)
        ax.scatter(x1, y1, label='leader', color='black')
        for i in range(self.n_followers):
            x_i = self.flap[i, 0]
            y_i = self.A_f[i] * numpy.sin(2 * numpy.pi * self.f_f[i] * self.tt)
            ax.scatter([x_i], [y_i], label=f'follower_{i+1}')
        ax.set_xlim([x1 - 50, x1 + 5])
        ax.set_ylim([-2*self.A1, 2*self.A1])
        ax.legend()
        fig.canvas.draw()
        frame = numpy.frombuffer(fig.canvas.tostring_rgb(), dtype=numpy.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return frame

    def close(self):
        pass