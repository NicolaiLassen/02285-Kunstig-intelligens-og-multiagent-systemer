from random import randint
from typing import List, Dict, Union

import torch
import torch.nn.functional as F
from machin.frame.buffers import Buffer
from machin.frame.transition import Transition
from torch import nn

from environment.env_wrapper import MultiAgentEnvWrapper

# ref: https://github.com/iffiX/machin/blob/d10727b52d981c898e31cdd20b48a3d972612bb6/machin/frame/algorithms/maddpg.py#L500
# Multi-Agent Actor-Critic for Mixed Cooperative Environments
class SHMBuffer(Buffer):
    @staticmethod
    def make_tensor_from_batch(batch, device, concatenate):
        if concatenate and len(batch) != 0:
            item = batch[0]
            batch_size = len(batch)
            if torch.is_tensor(item):
                batch = [it.to(device) for it in batch]
                result = torch.cat(batch, dim=0).to(device)
                result.share_memory_()
                return result
            else:
                try:
                    result = torch.tensor(batch, device=device).view(batch_size, -1)
                    result.share_memory_()
                    return result
                except Exception:
                    raise ValueError(f"Batch not concatenable: {batch}")
        else:
            for it in batch:
                if torch.is_tensor(it):
                    it.share_memory_()
            return batch


# https://arxiv.org/pdf/1706.02275.pdf
# ICM, Curiosity-driven Exploration by Self-supervised Prediction
# ref: https://github.com/iffiX/machin/blob/d10727b52d981c898e31cdd20b48a3d972612bb6/machin/frame/algorithms/maddpg.py#L500
class MADDPG():
    replay_buffers = []

    def __init__(self,
                 env: MultiAgentEnvWrapper,
                 actors: List[nn.Module],
                 actor_targets: List[nn.Module],
                 critics: List[nn.Module],
                 critic_targets: List[nn.Module],
                 targets: List[Union[NeuralNetworkModule, nn.Module]],
                 icm: nn.Module,
                 optimizer
                 ):
        self.env = env
        self.icm = icm  # shared ICM for state explore push
        self.actors = actors

    def train(self, max_time_steps: int, horizon: int, device: str):

        self.replay_buffers = [SHMBuffer(horizon, device) for _ in range(len(self.actors))]

        t = 0
        s1 = self.env.reset()
        while t < max_time_steps:
            for ep_T in range(horizon):
                t += 1
                s = s1
                action, probs, log_prob = self.act(s)
                s1, r, d, _ = self.env.step(action)

                if t % horizon == 0:
                    self.__update()

    def act(self, states: Dict):
        print()

    def __criticize(self):
        print()

    def __update(self,
                 update_value=True,
                 update_policy=True,
                 update_target=True,
                 concatenate_samples=True,
                 ):
        """
        Update network weights by sampling from replay buffer.
        Args:
            update_value: Whether to update the Q network.
            update_policy: Whether to update the actor network.
            update_target: Whether to update targets.
            concatenate_samples: Whether to concatenate the samples.
        Returns:
            mean value of estimated policy value, value loss
        """
        # All buffers should have the same length now.

        # Create a sample method per update
        # this sample method will sample the same indexes
        # (different for each update() call) on all buffers.

        buffer_length = self.replay_buffers[0].size()
        if buffer_length == 0:
            return
        batch_size = min(buffer_length, self.batch_size)
        sample_indexes = [
            [randint(0, buffer_length - 1) for _ in range(batch_size)]
            for __ in range(self.ensemble_size)
        ]

        sample_methods = [
            self._create_sample_method(indexes) for indexes in sample_indexes
        ]

        # Now sample from buffer for each sub-policy in the ensemble.
        # To reduce memory usage, for each sub-policy "i" of each actor,
        # the same sample "i" will be used for training.

        # Tensors in the sampled batch will be moved to shared memory.

        # size: [ensemble size, num of actors]
        batches = []
        next_actions_t = []
        for e_idx in range(self.ensemble_size):
            ensemble_batch = []
            for a_idx in range(len(self.actors)):
                batch_size_, batch = self.replay_buffers[a_idx].sample_batch(
                    self.batch_size,
                    concatenate_samples,
                    sample_method=sample_methods[e_idx],
                    sample_attrs=[
                        "state",
                        "action",
                        "reward",
                        "next_state",
                        "terminal",
                        "*",
                    ],
                )
                ensemble_batch.append(batch)
                assert batch_size_ == batch_size

            batches.append(ensemble_batch)
            next_actions_t.append(
                [
                    self.action_transform_function(act)
                    for act in self.act([batch[3] for batch in ensemble_batch], target=True)
                ]
            )

        if self.pool_type == "process":
            batches = self._move_to_shared_mem(batches)
            next_actions_t = self._move_to_shared_mem(next_actions_t)

        args = []
        self._update_counter += 1
        for e_idx in range(self.ensemble_size):
            for a_idx in range(len(self.actors)):
                args.append(
                    (
                        batch_size,
                        batches,
                        next_actions_t,
                        a_idx,
                        e_idx,
                        self.actors,
                        self.actor_targets,
                        self.critics,
                        self.critic_targets,
                        self.critic_visible_actors,
                        self.actor_optims,
                        self.critic_optims,
                        update_value,
                        update_policy,
                        update_target,
                        self.action_transform_function,
                        self.action_concat_function,
                        self.state_concat_function,
                        self.reward_function,
                        self.criterion,
                        self.discount,
                        self.update_rate,
                        self.update_steps,
                        self._update_counter,
                        self.grad_max,
                        self.visualize and not self.has_visualized,
                        self.visualize_dir,
                        self._backward,
                    )
                )

        all_loss = self.pool.starmap(self._update_sub_policy, args)
        mean_loss = torch.tensor(all_loss).mean(dim=0)

        # returns action value and policy loss
        return -mean_loss[0].item(), mean_loss[1].item()

    def store_transitions(self, transitions: List[Union[Transition, Dict]]):
        """
        Add a list of transition samples, from all actors at the same time
        step, to the replay buffers.
        Args:
            transitions: List of transition objects.
        """
        assert len(transitions) == len(self.replay_buffers)
        for buff, trans in zip(self.replay_buffers, transitions):
            buff.append(
                trans,
                required_attrs=("state", "action", "next_state", "reward", "terminal"),
            )

    def __intrinsic_reward_objective(self):
        states = self.mem_buffer.map_states.unsqueeze(1)
        next_states = self.mem_buffer.map_next_states.unsqueeze(1)
        action_probs = self.mem_buffer.action_probs
        actions = self.mem_buffer.actions

        a_t_hats, phi_t1_hats, phi_t1s, phi_ts = self.icm(states, next_states, action_probs)
        r_i_ts = F.mse_loss(phi_t1_hats, phi_t1s, reduction='none').sum(-1)

        return (self.eta * r_i_ts).detach(), r_i_ts.mean(), F.cross_entropy(a_t_hats, actions)
