# Plot the policy results saved in .csv file

import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy

class Logger:
    @staticmethod
    def plot_training(path, agent_idx=None, n_followers=None, sequential=False):
        """
        Plot training curves from either a single CSV file or by discovering agent
        logs within a folder.

        - If 'path' is a file: plots that single CSV (original behavior).
        - If 'path' is a folder and (n_followers and/or agent_idx) is provided:
          discovers the most recent CSV per requested agent and overlays them.

        Parameters
        - path: CSV file path or folder path
        - agent_idx: None, int, or list/tuple of ints. If provided, only those agents are plotted.
        - n_followers: None or int. If provided, restricts discovered logs to this N.
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

        # Case 1: direct CSV path
        if os.path.isfile(path):
            df = pd.read_csv(path)
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df['Episode'], df['reward'], label='Reward')
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Reward')
            ax.tick_params(axis='y')
            ax.grid(False)
            return fig

        # Case 2: discover within folder based on agent/N filters
        if not os.path.isdir(path):
            raise ValueError(f"Path does not exist: {path}")

        if agent_idx is None and (n_followers is None and not sequential):
            raise ValueError("When providing a folder, specify n_followers and/or agent_idx to select which curves to plot.")

        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        # Filter by naming scheme
        if sequential:
            csv_files = [f for f in csv_files if "_SEQ_AG" in f]
        elif n_followers is not None:
            csv_files = [f for f in csv_files if f"_N{n_followers}_AG" in f or f"Chain_N{n_followers}_AG" in f]

        # Normalize agent_idx into a list (or discover all agents for given N)
        agent_indices_to_plot = None
        if agent_idx is None:
            # Derive agents from filenames for this N
            agent_indices_to_plot = []
            # Filenames use 1-based AG ids
            if sequential:
                pattern = re.compile(r"_SEQ_AG(\d+)_")
            else:
                pattern = re.compile(rf"_N{n_followers}_AG(\d+)_") if n_followers is not None else re.compile(r"_AG(\d+)_")
            for f in csv_files:
                m = pattern.search(f)
                if m:
                    agent_num = int(m.group(1))
                    if agent_num not in agent_indices_to_plot:
                        agent_indices_to_plot.append(agent_num)
            if not agent_indices_to_plot:
                raise ValueError("No agent logs found in the folder for the specified filters.")
        else:
            if isinstance(agent_idx, int):
                if agent_idx < 1:
                    raise ValueError("agent_idx must be 1-based (>=1)")
                agent_indices_to_plot = [agent_idx]
            elif isinstance(agent_idx, (list, tuple)):
                agent_indices_to_plot = []
                for a in agent_idx:
                    if not isinstance(a, int) or a < 1:
                        raise ValueError("agent_idx list must contain 1-based integers")
                    agent_indices_to_plot.append(a)
            else:
                raise TypeError("agent_idx must be an int, list, or tuple of ints")

        # For each requested agent, pick most recent file and plot
        fig, ax = plt.subplots(figsize=(8, 4))
        for idx in sorted(agent_indices_to_plot):
            # Match 1-based AG ids in filenames
            if sequential:
                matching = [f for f in csv_files if f"_SEQ_AG{idx}_" in f]
            else:
                matching = [f for f in csv_files if f"_AG{idx}_" in f]
            if not matching:
                raise ValueError(f"No CSV files found for agent {idx} under the specified filters.")
            matching.sort(key=lambda f: os.path.getmtime(os.path.join(path, f)), reverse=True)
            csv_path = os.path.join(path, matching[0])
            df = pd.read_csv(csv_path)
            ax.plot(df['Episode'], df['reward'], label=f"Agent {idx}")

        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward')
        ax.legend(loc='lower right')
        ax.grid(False)
        return fig

    # Plot average of many training runs (optionally filter by agent or substring)
    @staticmethod
    def plot_average_training(folder_path, agent_idx=None, filename_contains=None, n_followers=None, sequential=False):
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
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder does not exist: {folder_path}")
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        # Filter by naming scheme
        if sequential:
            csv_files = [f for f in csv_files if "_SEQ_AG" in f]
        if agent_idx is not None:
            # Accept int or list; filenames use 1-based AG ids
            if isinstance(agent_idx, int):
                csv_files = [f for f in csv_files if f"_AG{agent_idx}_" in f]
            else:
                wanted = set(int(x) for x in (agent_idx if isinstance(agent_idx, (list, tuple)) else [agent_idx]))
                csv_files = [f for f in csv_files if any(f"_AG{aid}_" in f for aid in wanted)]
        if filename_contains:
            csv_files = [f for f in csv_files if filename_contains in f]
        if (n_followers is not None) and (not sequential):
            csv_files = [
                f for f in csv_files
                if f"_N{n_followers}_AG" in f or f"Chain_N{n_followers}_AG" in f
            ]
        num_runs = len(csv_files)

        if num_runs == 0:
            raise ValueError("No CSV files found in the specified folder (after filtering).")
        
        # Read all runs into a list of DataFrames
        dfs = []
        for csv_file in csv_files:
            csv_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(csv_path)
            dfs.append(df)
        
        # Stack rewards for all runs
        rewards_matrix = numpy.stack([df['reward'].values for df in dfs])
        episodes = dfs[0]['Episode'].values
        avg_reward = numpy.mean(rewards_matrix, axis=0)
        std_reward = numpy.std(rewards_matrix, axis=0)
        upper = avg_reward + std_reward
        lower = avg_reward - std_reward

        # Plot average and shaded std deviation with gradient
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(episodes, avg_reward, color='black', linewidth=2, label='Average Reward')

        # Solid gray shading for standard deviation
        ax.fill_between(episodes, lower, upper, color='gray', alpha=0.3, linewidth=0)

        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward')
        ax.legend(loc='lower right')

        return fig