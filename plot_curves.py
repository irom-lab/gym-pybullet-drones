import sys
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import csv
import torch
import pretty_errors


def check_train(trial_name):

    filename = 'logs/'+trial_name+'/evaluations.npz'
    data = np.load(filename)
    eval_timestep_all = []
    eval_mean_reward_all = []
    for j in range(data['timesteps'].shape[0]):
        print(str(data['timesteps'][j])+","+str(data['results'][j][0]))
        eval_timestep_all += [data['timesteps'][j]]
        eval_mean_reward_all += [data['results'][j][0]]

    ##################### Load data ######################

    # Load training details
    progress_path = 'logs/'+trial_name+'/progress.csv'

# time/total timesteps,train/ent_coef,time/fps,train/critic_loss,rollout/ep_len_mean,train/learning_rate,train/n_updates,time/time_elapsed,rollout/ep_rew_mean,train/ent_coef_loss,train/actor_loss,time/episodes,eval/mean_reward,eval/mean_ep_length

# rollout/ep_len_mean,time/fps,time/time_elapsed,time/iterations,rollout/ep_rew_mean,time/total_timesteps,train/std,train/loss,train/explained_variance,train/n_updates,train/approx_kl,train/value_loss,train/policy_gradient_loss,train/entropy_loss,train/clip_fraction,train/clip_range,train/learning_rate,eval/mean_reward,time/total timesteps,eval/mean_ep_length

    # entries: ReturnAverage, lossAverage
    timestep_all = []
    critic_loss_all = []
    actor_loss_all = []
    alpha_all = []
    entropy_all = []
    reader = csv.DictReader(open(progress_path))
    for itr, r in enumerate(reader):
        if r['time/total_timesteps'] != '':
            timestep_all += [float(r['time/total_timesteps'])]
        else:
            timestep_all += [timestep_all[-1]]
        # if r['train/critic_loss'] != '':
            # critic_loss_all += [float(r['train/critic_loss'])]
        # else:
        #     critic_loss_all += [critic_loss_all[-1]]
        # if r['train/actor_loss'] != '':
        #     actor_loss_all += [float(r['train/actor_loss'])]
        # else:
        #     actor_loss_all += [actor_loss_all[-1]]
        # if r['train/ent_coef'] != '':
        #     alpha_all += [float(r['train/ent_coef'])]
        # else:
        #     alpha_all += [alpha_all[-1]]
        # if r['train/entropy'] != '':
        #     entropy_all += [float(r['train/entropy'])]
        # else:
        #     entropy_all += [entropy_all[-1]]

    print('Max reward: ', np.max(eval_mean_reward_all))
    print('Step: ', [eval_timestep_all[np.argmax(eval_mean_reward_all)]])

    #################   Success    #################

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
    ax1.plot(eval_timestep_all, eval_mean_reward_all, color='brown', label='Avg return')
    # ax2.plot(timestep_all, critic_loss_all, color='brown', label='Avg q loss')
    # ax3.plot(timestep_all, actor_loss_all, color='brown', label='Avg pi loss')
    # ax4.plot(timestep_all, alpha_all, color='brown', label='Alpha')
    # ax5.plot(timestep_all, entropy_all, color='brown', label='Entropy')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    plt.show()


if __name__ == '__main__':
    yaml_file_name = sys.argv[1]
    check_train(trial_name=yaml_file_name)
