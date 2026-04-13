
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist


class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class A3CAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, is_discrete=False):
        super().__init__()
        self.is_discrete = is_discrete

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )

        if not self.is_discrete:
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        value = self.critic(x)
        mean = self.actor_mean(x)

        if self.is_discrete:
            distrib = dist.Categorical(logits=mean)
        else:
            std = self.actor_logstd.expand_as(mean).exp()
            distrib = dist.Normal(mean, std)

        if action is None:
            action = distrib.sample()

        log_prob = distrib.log_prob(action)
        if not self.is_discrete:
            log_prob = log_prob.sum(dim=-1)

        entropy = distrib.entropy()
        if not self.is_discrete:
            entropy = entropy.sum(dim=-1)

        return action, log_prob, entropy, value


class A2CAgent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.is_discrete = isinstance(envs.action_space, gym.spaces.Discrete)
        obs_dim = envs.observation_space.shape[0]

        if self.is_discrete:
            act_dim = envs.action_space.n
        else:
            act_dim = envs.action_space.shape[0]

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim)
        )

        if not self.is_discrete:
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        if self.is_discrete:
            distrib = dist.Categorical(logits=mean)
        else:
            action_logstd = self.actor_logstd.expand_as(mean)
            action_std = torch.exp(action_logstd)
            distrib = dist.Normal(mean, action_std)

        if action is None:
            action = distrib.sample()

        if self.is_discrete:
            log_prob = distrib.log_prob(action)
            entropy = distrib.entropy()
        else:
            log_prob = distrib.log_prob(action).sum(1)
            entropy = distrib.entropy().sum(1)

        return action, log_prob, entropy, self.critic(x)


class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, is_discrete=False,
                 action_high=None, action_low=None):
        super().__init__()
        self.is_discrete = is_discrete

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, act_dim)
        )

        if not self.is_discrete:
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        if self.is_discrete:
            distrib = torch.distributions.Categorical(logits=mean)
        else:
            action_logstd = self.actor_logstd.expand_as(mean)
            action_std = torch.exp(action_logstd)
            distrib = torch.distributions.Normal(mean, action_std)

        if action is None:
            action = distrib.sample()

        if self.is_discrete:
            log_prob = distrib.log_prob(action)
            entropy = distrib.entropy()
        else:
            log_prob = distrib.log_prob(action).sum(1)
            entropy = distrib.entropy().sum(1)

        return action, log_prob, entropy, self.critic(x)


def a3c_worker(worker_id, global_agent, global_optimizer, grad_lock,
               env_id, seed, num_episodes,
               gamma, num_steps, ent_coef, vf_coef, max_grad_norm,
               result_queue):
    env = gym.make(env_id)
    env.action_space.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    local_agent = A3CAgent(obs_dim, act_dim, is_discrete=False)

    for episode in range(num_episodes):
        local_agent.load_state_dict(global_agent.state_dict())
        state, _ = env.reset(seed=seed + worker_id + episode)
        done = False
        ep_reward = 0.0

        while not done:
            rewards, values, log_probs, entropies = [], [], [], []

            for _ in range(num_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, log_prob, entropy, value = local_agent.get_action_and_value(state_tensor)

                action_np = action.squeeze(0).detach().numpy()
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

                next_state, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropies.append(entropy)

                ep_reward += reward
                state = next_state

                if done:
                    break

            R = torch.zeros(1, 1)
            if not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    R = local_agent.get_value(state_tensor)

            policy_loss = 0.0
            value_loss = 0.0
            entropy_loss = 0.0

            for i in reversed(range(len(rewards))):
                R = rewards[i] + gamma * R
                advantage = R - values[i]
                value_loss = value_loss + advantage.pow(2)
                policy_loss = policy_loss - log_probs[i] * advantage.detach()
                entropy_loss = entropy_loss - entropies[i]

            total_loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
            total_loss = total_loss.mean()

            local_agent.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(local_agent.parameters(), max_grad_norm)

            with grad_lock:
                global_optimizer.zero_grad()
                for local_param, global_param in zip(local_agent.parameters(), global_agent.parameters()):
                    global_param._grad = local_param.grad
                global_optimizer.step()

            local_agent.load_state_dict(global_agent.state_dict())

        result_queue.put((worker_id, episode, float(ep_reward)))

    env.close()


def a2c_worker(worker_id, global_agent,
               env_id, seed, num_steps, num_updates,
               shared_obs, shared_actions, shared_logprobs,
               shared_rewards, shared_dones, shared_values,
               shared_next_obs, shared_next_done,
               collect_barrier, update_barrier, ep_queue):
    env = gym.make(env_id)
    env.action_space.seed(seed + worker_id)
    obs, _ = env.reset(seed=seed + worker_id)
    done = False

    for update in range(num_updates):
        for step in range(num_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, logprob, _, value = global_agent.get_action_and_value(obs_tensor)

            action_np = action.squeeze(0).numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            shared_obs[step, worker_id] = obs_tensor.squeeze(0)
            shared_actions[step, worker_id] = action.squeeze(0)
            shared_logprobs[step, worker_id] = logprob.squeeze(0)
            shared_rewards[step, worker_id] = reward
            shared_dones[step, worker_id] = done
            shared_values[step, worker_id] = value.squeeze(0)

            obs = next_obs

            if done:
                obs, _ = env.reset()
                ep_queue.put((update, reward))

        shared_next_obs[worker_id] = torch.FloatTensor(obs)
        shared_next_done[worker_id] = done

        collect_barrier.wait()
        update_barrier.wait()

    env.close()


def ppo_worker(worker_id, global_agent,
               env_id, seed, num_steps, num_updates,
               shared_obs, shared_actions, shared_logprobs,
               shared_rewards, shared_dones, shared_values,
               shared_next_obs, shared_next_done,
               collect_barrier, update_barrier, ep_queue):
    env = gym.make(env_id)
    env.action_space.seed(seed + worker_id)
    obs, _ = env.reset(seed=seed + worker_id)
    done = False
    ep_return = 0.0

    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if not is_discrete:
        action_low = torch.tensor(env.action_space.low, dtype=torch.float32)
        action_high = torch.tensor(env.action_space.high, dtype=torch.float32)

    for update in range(num_updates):
        for step in range(num_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _, _, value = global_agent.get_action_and_value(obs_tensor)

                if is_discrete:
                    action_to_store = action.long()
                    action_env = int(action_to_store.item())
                else:
                    action_to_store = torch.max(
                        torch.min(action, action_high.unsqueeze(0)),
                        action_low.unsqueeze(0)
                    )
                    action_env = action_to_store.squeeze(0).numpy()

                _, logprob, _, value = global_agent.get_action_and_value(obs_tensor, action_to_store)

            next_obs, reward, terminated, truncated, _ = env.step(action_env)
            done = terminated or truncated
            ep_return += reward

            shared_obs[step, worker_id] = obs_tensor.squeeze(0)
            shared_actions[step, worker_id] = action_to_store.squeeze(0)
            shared_logprobs[step, worker_id] = logprob.squeeze(0)
            shared_rewards[step, worker_id] = reward
            shared_dones[step, worker_id] = done
            shared_values[step, worker_id] = value.squeeze(0)

            obs = next_obs

            if done:
                ep_queue.put((update, float(ep_return)))
                obs, _ = env.reset()
                ep_return = 0.0

        shared_next_obs[worker_id] = torch.FloatTensor(obs)
        shared_next_done[worker_id] = done

        collect_barrier.wait()
        update_barrier.wait()

    env.close()
