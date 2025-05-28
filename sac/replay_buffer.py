import abc

class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    Abstract base class for replay buffers.
    """
    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):

        """
        Add a single sample to the replay buffer.
        Args:
            obs: Observation from the environment.
            action: Action taken by the agent.
            reward: Reward received from the environment.
            next_obs: Next observation after taking the action.
            terminal: Boolean indicating if the episode has ended.
            agent_info: Additional information from the agent.
            env_info: Additional information from the environment.
        """
        pass
    @abc.abstractmethod
    def terminate_episode(self):
        """
        Mark the end of an episode in the replay buffer.
        This can be used to handle any necessary cleanup or state updates.
        """
        pass

    @property
    @abc.abstractmethod
    def size(self, **kwargs):
        """
        Return the current size of the replay buffer.
        """
        pass


    def add_path(self, path):
        """
        Add a complete path (sequence of transitions) to the replay buffer.
        """
        
        for i, (
            obs,
            action, 
            reward, 
            next_obs, 
            terminal, 
            agent_info, 
            env_info 
            ) in enumerate(zip(
                path['observations'],
                path['actions'],
                path['rewards'],
                path['next_observations'],
                path['terminals'],
                path['agent_infos'],
                path['env_infos']
            )):
                self.add_sample(
                    obs=obs,
                    action=action,
                    reward=reward,
                    next_obs=next_obs,
                    terminal=terminal,
                    agent_info=agent_info,
                    env_info=env_info
                )
        self.terminate_episode()
    
    @abc.abstractmethod
    def random_batch(self, batch_size):
        """
        Return a batch of size `batch_size`.
        :param batch_size:
        :return:
        """
        pass