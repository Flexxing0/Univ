from IPython.core import application
from collections import defaultdict
from tqdm import tqdm
from matplotlib import pyplot as plt
import gymnasium as gym 
import numpy as np  

class Q_learning:
    
    def __init__(self, env = gym.make('FrozenLake-v1', is_slippery=True), alpha = 0.1, episodes= 10000, gamma = 0.99, epsilon= 1.0, epsilon_decay = 0.999, final_epsilon = 0.01):
        self.env = env
        
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        
        self.alpha = alpha #learning rate(que tan rapido actualiza q-values(0-1))
        self.episodes = episodes #cantidad episodios de entrenamiento
        self.gamma = gamma #discount factor(descuento de recompensa(0-1))
        self.epsilon = epsilon #exploration rate(chance de explorar(0-1))
        self.epsilon_decay = epsilon_decay #decay rate (decremento de epsilon(0-1))
        self.final_epsilon = final_epsilon
        
        self.training_error = []
    
    def get_action(self, obs: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
    
    def update(self, obs:tuple[int, int, bool], action: int, reward: float, terminated: bool, next_obs: tuple[int,int, bool]):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.gamma * future_q_value #ecuacion de bellman
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] = (self.q_values[obs][action] + self.alpha * temporal_difference)
        
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        
    def entrenamiento(self):
        for episode in tqdm(range(self.episodes)):
            obs, info = self.env.reset()
            done = False
            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.update(obs, action, reward, terminated, next_obs)
                
                done = terminated or truncated
                obs = next_obs
                
            self.decay_epsilon()
            
    def resultados(self):
        self.entrenamiento()
        rolling_length = 500
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

        # Episode rewards (win/loss performance)
        axs[0].set_title("Episode rewards")
        reward_moving_average = self._get_moving_avgs(
            self.env.return_queue,
            rolling_length,
            "valid"
        )
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[0].set_ylabel("Average Reward")
        axs[0].set_xlabel("Episode")

        # Episode lengths (how many actions per hand)
        axs[1].set_title("Episode lengths")
        length_moving_average = self._get_moving_avgs(
            self.env.length_queue,
            rolling_length,
            "valid"
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[1].set_ylabel("Average Episode Length")
        axs[1].set_xlabel("Episode")

        # Training error (how much we're still learning)
        axs[2].set_title("Training Error")
        training_error_moving_average = self._get_moving_avgs(
            self.training_error,
            rolling_length,
            "same"
        )
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
        axs[2].set_ylabel("Temporal Difference Error")
        axs[2].set_xlabel("Step")

        plt.tight_layout()
        plt.show()
        
        
    def _get_moving_avgs(self, arr, window, convolution_mode):
        """Compute moving average to smooth noisy data."""
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
    ) / window
        
    def test_agent(self, env, num_episodes=1000):
        """Test agent performance without learning or exploration."""
        total_rewards = []

        # Temporarily disable exploration for testing
        old_epsilon = self.epsilon
        self.epsilon = 0.0  # Pure exploitation

        for _ in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action = self.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)

        # Restore original epsilon
        self.epsilon = old_epsilon

        win_rate = np.mean(np.array(total_rewards) > 0)
        average_reward = np.mean(total_rewards)

        print(f"Test Results over {num_episodes} episodes:")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Average Reward: {average_reward:.3f}")
        print(f"Standard Deviation: {np.std(total_rewards):.3f}")
        

if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=10000)
    agent = Q_learning(env)
    agent.resultados()
    agent.test_agent(env)