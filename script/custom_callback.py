from stable_baselines3.common.callbacks import BaseCallback
import wandb


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        if hasattr(self.logger, 'train_rollout_record_copy'):
            step = self.logger.train_rollout_record_copy['time/total_timesteps']
            if hasattr(self.logger, 'eval_rollout_record_copy'):
                out = {
                    'eval reward': self.logger.eval_rollout_record_copy['eval/mean_reward'],
                    'eval length': self.logger.eval_rollout_record_copy['eval/mean_ep_length'],
                }
                wandb.log(out, step=step, commit=False)
            out = {
                'episodes': self.logger.train_rollout_record_copy['time/episodes'],
                'fps': self.logger.train_rollout_record_copy['time/fps'],
                'train reward': self.logger.train_rollout_record_copy['rollout/ep_rew_mean'],
                'train length': self.logger.train_rollout_record_copy['rollout/ep_len_mean'],
                #
                'sac_ent_coef': self.logger.train_record_copy['train/ent_coef'],
                'sac_actor_loss': self.logger.train_record_copy['train/actor_loss'],
                'sac_critic_loss': self.logger.train_record_copy['train/critic_loss'],
                'sac_entropy': self.logger.train_record_copy['train/entropy'],
                'sac_num_update': self.logger.train_record_copy['train/n_updates'],
            }
            
            wandb.log(out, step=step, commit=True)
