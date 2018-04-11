import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


def log_run(agent, file_name, given_labels = None):
    if(given_labels):
        labels = given_labels
    else:
        labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
                'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
                'psi_velocity', 'action', 'reward', 'done']
    results = {x: [] for x in labels}

    # Run the simulation, and save the results.
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        state = agent.reset_episode()
        while True:
            action = agent.act(state) # rotor_speeds
            next_state, reward, done = agent.task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            to_write = [agent.task.sim.time]
            to_write += list(agent.task.sim.pose)
            to_write += list(agent.task.sim.v)
            to_write += list(agent.task.sim.angular_v)
            to_write += [action]
            to_write += [reward, done]
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)
            if done:
                break
        print("final_reward: ", reward)
    return results


def load_log(file_path):
    return pd.read_csv(file_path)


def plot_log(file_path):
    results = load_log(file_path)
    plot_run(results)

def normalize_angle(angles):
    # Adjust angles to range -pi to pi
    norm_angles = np.copy(angles)
    for i in range(len(norm_angles)):
        while norm_angles[i] > np.pi:
            norm_angles[i] -= 2 * np.pi
    return norm_angles


def plot_z_n_reward(results):
    plt.plot(results['time'], results['z'], label='z-pos')
    plt.plot(results['time'], results['reward'], label='reward')
    plt.xlabel('time, seconds')
    plt.ylabel('z-Position, Reward')
    plt.legend()
    _ = plt.ylim()

def plot_run(results):
    plt.subplots(figsize=(15, 15))

    plt.subplot(3, 3, 1)
    plt.title('Position')
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.xlabel('time, seconds')
    plt.ylabel('Position')
    plt.legend()

    plt.subplot(3, 3, 2)
    plt.title('Velocity')
    plt.plot(results['time'], results['x_velocity'], label='x')
    plt.plot(results['time'], results['y_velocity'], label='y')
    plt.plot(results['time'], results['z_velocity'], label='z')
    plt.xlabel('time, seconds')
    plt.ylabel('Velocity')
    plt.legend()

    plt.subplot(3, 3, 3)
    plt.title('Orientation')
    plt.plot(results['time'], normalize_angle(results['phi']), label='phi')
    plt.plot(results['time'], normalize_angle(results['theta']), label='theta')
    plt.plot(results['time'], normalize_angle(results['psi']), label='psi')
    plt.xlabel('time, seconds')
    plt.legend()

    plt.subplot(3, 3, 4)
    plt.title('Angular Velocity')
    plt.plot(results['time'], results['phi_velocity'], label='phi')
    plt.plot(results['time'], results['theta_velocity'], label='theta')
    plt.plot(results['time'], results['psi_velocity'], label='psi')
    plt.xlabel('time, seconds')
    plt.legend()

    plt.subplot(3, 3, 5)
    plt.title('Rotor Speed')
    plt.plot(results['time'], results['action'], label='Action')
    plt.xlabel('time, seconds')
    plt.ylabel('Rotor Speed, revolutions / second')
    plt.legend()

    plt.subplot(3, 3, 6)
    plt.title('Reward')
    plt.plot(results['time'], results['reward'], label='Reward')
    plt.xlabel('time, seconds')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot runtime for reward and z-position over episodes
def plt_dynamic(fig1, fig2, x, y1, sub1, y2, sub2, color_y1='g', color_y2='b'):
    # fig, sub1 = plt.subplots(1,1)
    sub1.plot(x, y1, color_y1)
    sub2.plot(x, y2, color_y2)
    fig1.canvas.draw()
    fig2.canvas.draw()

def createPlot(num_episodes):
# fig and sub1, sub2
    fig1, sub1 = plt.subplots(1,1)
    fig2, sub2 = plt.subplots(1,1)
    # sub2 = sub1.twinx()

    # labes
    sub1.set_xlabel('Episode')
    sub1.set_ylabel('Reward')
    sub1.tick_params(axis='x', colors='k')
    sub1.tick_params(axis='y', colors="g")
    sub2.set_xlabel('Episode')
    sub2.set_ylabel('Z')
    sub2.tick_params(axis='y', colors='b')

    #boundaries
    sub1.set_xlim(1,num_episodes)
    sub1.set_ylim(-0.0, 50.0)
    sub2.set_xlim(1, num_episodes)
    sub2.set_ylim(0.0, 200.0)

    plt.legend()
    return fig1, fig2, sub1, sub2
