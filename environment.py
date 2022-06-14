import numpy as np
import APF_function_for_DQN
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import escaper


class environment():
    def __init__(self, gamma):
        self.num_agent = 3
        self.num_action = 24
        self.num_state = 4 + 1 + (self.num_agent - 1) * 2
        self.t = 0
        self.v = 300
        self.delta_t = 0.1
        self.wall_following = np.zeros((1, self.num_agent))
        self.gamma = gamma
        self.r_perception = 2000

    def reset(self):
        self.t = 0
        self.boundary = APF_function_for_DQN.generate_boundary(np.array([[0.0], [0]]), np.array([[3600], [0]]),
                                                               np.array([[3600], [5000]]), np.array([[0], [5000]]), 51)

        self.obstacle1 = APF_function_for_DQN.generate_boundary(np.array([[900.0], [1000]]),
                                                                np.array([[1550], [1000]]),
                                                                np.array([[1550], [1100]]), np.array([[900], [1100]]),
                                                                11)
        self.obstacle2 = APF_function_for_DQN.generate_boundary(np.array([[2050.0], [1000]]),
                                                                np.array([[2700], [1000]]),
                                                                np.array([[2700], [1100]]), np.array([[2050], [1100]]),
                                                                11)
        self.obstacle3 = APF_function_for_DQN.generate_boundary(np.array([[1400.0], [2450]]),
                                                                np.array([[2200], [2450]]),
                                                                np.array([[2200], [2550]]), np.array([[1400], [2550]]),
                                                                11)
        self.obstacle4 = APF_function_for_DQN.generate_boundary(np.array([[900.0], [3900]]),
                                                                np.array([[1550], [3900]]),
                                                                np.array([[1550], [4000]]), np.array([[900], [4000]]),
                                                                11)
        self.obstacle5 = APF_function_for_DQN.generate_boundary(np.array([[2050.0], [3900]]),
                                                                np.array([[2700], [3900]]),
                                                                np.array([[2700], [4000]]), np.array([[2050], [4000]]),
                                                                11)

        self.obstacle_total = np.hstack((self.boundary, self.obstacle1, self.obstacle2, self.obstacle3, self.obstacle4,
                                         self.obstacle5))
        self.obstacle_with_other_agent = self.obstacle_total

        self.target_position = np.random.random((2, 1))

        self.target_position[0] = self.target_position[0] * 3200 + 200
        self.target_position[1] = self.target_position[1] * 600 + 4200
        self.target_orientation = np.array([[0.], [1]])
        self.zigzag_count = 0
        self.zigzag_last = np.zeros((2, 1))
        self.escaper_zigzag_flag = 0
        self.escaper_wall_following = 0
        self.escaper_slip_flag = 0
        self.last_e = np.zeros((2, 1))

        # self.target_position = np.array([[1000], [3000]])

        self.agent_position = np.random.random((2, 1))
        self.agent_position[0, :] = self.agent_position[0, :] * 2400 + 200
        self.agent_position[1, :] = self.agent_position[1, :] * 600 + 200
        self.agent_position = self.agent_position.repeat(3, axis=1) + np.array([[0, 400, 800], [0, 0, 0]])

        # self.agent_position = np.array([[1800],[500]])
        self.agent_position_origin = self.agent_position

        self.done = np.zeros((1, self.num_agent))

        self.agent_orientation = np.vstack((np.zeros((1, self.num_agent)), np.ones((1, self.num_agent))))
        self.agent_orientation_origin = self.agent_orientation

        self.update_state()

        return self.state

    def reward(self):
        reward = np.zeros((1, self.num_agent))
        done = np.zeros((1, self.num_agent))
        position_buffer = self.agent_position.copy()
        success_flag = np.any(self.done)
        self.distance_from_target = np.linalg.norm(self.agent_position - self.target_position, axis=0, keepdims=True)

        for i in range(self.num_agent):
            reward2 = 0
            reward3 = 0
            reward4 = 0
            reward5 = 0
            if success_flag:
                success_range = 300
            else:
                success_range = 200
            if np.linalg.norm(self.agent_position[:, i:i + 1] - self.target_position) < success_range:
                reward1 = 20
                done_temp = 1.
            else:
                reward1 = 0
                done_temp = 0.
                if np.arccos(np.clip(
                        np.dot(np.ravel(self.agent_orientation_last[:, i:i + 1]),
                               np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                            self.agent_orientation_last[:, i:i + 1]) / np.linalg.norm(
                            self.agent_orientation[:, i:i + 1]),
                        -1, 1)) > np.radians(45):
                    reward2 = -5
                else:
                    reward2 = 0

                temp = np.argmin(np.linalg.norm(self.obstacle_total - self.agent_position[:, i:i + 1], axis=0))
                self.obstacle_closest[:, i:i + 1] = self.obstacle_total[:, temp:temp + 1]
                if np.linalg.norm(self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]) > 150:
                    reward3 = 0
                elif np.linalg.norm(self.agent_position[:, i:i + 1] - self.obstacle_closest[:, i:i + 1]) < 100:
                    reward3 = -20
                    position_buffer[:, i:i + 1] = self.agent_position_origin[:, i:i + 1]
                else:
                    reward3 = -2

                if np.amin(np.linalg.norm(self.agent_position[:, i:i + 1] - np.delete(self.agent_position, i, axis=1),
                                          axis=0)) > 200:
                    reward4 = 0
                else:
                    reward4 = -20
                    position_buffer[:, i:i + 1] = self.agent_position_origin[:, i:i + 1]

                if self.distance_from_target[0, i] < 400:
                    potential = 15
                elif self.distance_from_target[0, i] < 600:
                    potential = 10
                elif self.distance_from_target[0, i] < 800:
                    potential = 5
                else:
                    potential = 0
                if self.distance_from_target_last[0, i] < 400:
                    potential_last = 15
                elif self.distance_from_target_last[0, i] < 600:
                    potential_last = 10
                elif self.distance_from_target_last[0, i] < 800:
                    potential_last = 5
                else:
                    potential_last = 0
                reward5 = self.gamma * potential - potential_last

                if self.t == 1000:
                    done_temp = 2.

            reward[0, i] = reward1 + reward2 + reward3 + reward4 + reward5
            done[0, i] = done_temp
        self.agent_position = position_buffer.copy()
        self.done = done.copy()

        return reward, done

    def step(self, action):
        self.agent_position_last = self.agent_position
        self.agent_orientation_last = self.agent_orientation
        self.distance_from_target_last = np.linalg.norm(self.agent_position - self.target_position, axis=0,
                                                        keepdims=True)
        self.t += 1
        ######agent#########
        scale_repulse = np.zeros((1, self.num_agent))
        individual_balance = np.zeros((1, self.num_agent))
        for i in range(self.num_agent):
            if action[0, i] < 8:
                scale_repulse[0, i] = 0
            elif action[0, i] < 16:
                scale_repulse[0, i] = 1.5e8
            else:
                scale_repulse[0, i] = 3e8
            if action[0, i] % 8 == 0:
                individual_balance[0, i] = 30
            elif action[0, i] % 8 == 1:
                individual_balance[0, i] = 100
            elif action[0, i] % 8 == 2:
                individual_balance[0, i] = 250
            elif action[0, i] % 8 == 3:
                individual_balance[0, i] = 500
            elif action[0, i] % 8 == 4:
                individual_balance[0, i] = 750
            elif action[0, i] % 8 == 5:
                individual_balance[0, i] = 1000
            elif action[0, i] % 8 == 6:
                individual_balance[0, i] = 2000
            elif action[0, i] % 8 == 7:
                individual_balance[0, i] = 3000

        F, self.wall_following = APF_function_for_DQN.total_decision(self.agent_position,
                                                                     self.agent_orientation,
                                                                     self.obstacle_closest_with_other_agent,
                                                                     self.target_position,
                                                                     scale_repulse,
                                                                     individual_balance,
                                                                     self.r_perception)
        F = np.round(F * 1000) / 1000
        agent_position_buffer = np.zeros((2, self.num_agent))
        for i in range(self.num_agent):
            if self.done[0, i]:
                agent_position_buffer[:, i:i + 1] = np.array([[0.], [0]])
            else:
                agent_position_buffer[:, i:i + 1] = F[:, i:i + 1] * self.v * self.delta_t

        #######escaper########
        F_escaper, zigzag_count, zigzag_last, escaper_zigzag_flag, escaper_wall_following, escaper_slip_flag, distance_from_nearest_obstacle, last_e = escaper.escaper(
            self.agent_position, self.target_position,
            self.target_orientation, self.obstacle_total,
            self.num_agent, self.zigzag_count, self.zigzag_last, self.last_e, self.escaper_slip_flag)
        self.zigzag_last = zigzag_last
        self.zigzag_count = zigzag_count
        self.escaper_zigzag_flag = escaper_zigzag_flag
        self.escaper_wall_following = escaper_wall_following
        self.escaper_slip_flag = escaper_slip_flag
        self.last_e = last_e
        #####update#####
        self.agent_position = self.agent_position + agent_position_buffer
        self.agent_orientation = F

        if np.any(self.done) or distance_from_nearest_obstacle < 30:
            pass
        else:
            self.target_position = self.target_position + F_escaper * self.delta_t
            self.target_orientation = F_escaper

        reward, done = self.reward()
        self.update_state()

        return self.state, reward, done

    def render(self):
        plt.figure(1)
        plt.cla()
        ax = plt.gca()
        plt.xlim([-100, 3700])
        plt.ylim([-100, 5100])
        ax.set_aspect(1)
        plt.plot(self.obstacle1[0, :], self.obstacle1[1, :], 'black')
        plt.plot(self.obstacle2[0, :], self.obstacle2[1, :], 'black')
        plt.plot(self.obstacle3[0, :], self.obstacle3[1, :], 'black')
        plt.plot(self.obstacle4[0, :], self.obstacle4[1, :], 'black')
        plt.plot(self.obstacle5[0, :], self.obstacle5[1, :], 'black')

        plt.plot(self.boundary[0, :], self.boundary[1, :], 'black')

        if self.escaper_slip_flag == 1:
            color = 'black'
        else:
            if self.escaper_wall_following == 1:
                color = 'green'
            else:
                if self.escaper_zigzag_flag == 1:
                    color = 'blue'
                else:
                    color = 'red'
        # plt.scatter(self.target_position[0, 0], self.target_position[1, 0], c=color)
        circle = mpatches.Circle(np.ravel(self.target_position), 100, facecolor=color)
        ax.add_patch(circle)

        color = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i in range(self.num_agent):
            # plt.scatter(self.agent_position[0, i], self.agent_position[1, i])
            circle = mpatches.Circle(self.agent_position[:, i], 100, facecolor=color[i])
            ax.add_patch(circle)
            if not self.done[0, i]:
                if self.wall_following[0, i]:
                    plt.quiver(self.agent_position[0, i], self.agent_position[1, i], self.agent_orientation[0, i],
                               self.agent_orientation[1, i], color='green', scale=10)
                else:
                    plt.quiver(self.agent_position[0, i], self.agent_position[1, i], self.agent_orientation[0, i],
                               self.agent_orientation[1, i], color='black', scale=10)
        # plt.title(str(float(self.state[0,0])) + ' ' + str(float(self.state[0,1])) + ' ' + str(float(self.state[0,2])))
        plt.show(block=False)
        # plt.savefig(str(self.t))
        plt.pause(0.001)

    def update_state(self):
        self.obstacle_closest = np.zeros((2, self.num_agent))
        self.state = np.zeros((self.num_state, self.num_agent))
        self.obstacle_closest_with_other_agent = np.zeros((2, self.num_agent))

        virtual_obstacle = np.zeros((2, 0))
        for i in range(self.num_agent):
            if self.done[0, i]:
                virtual_obstacle = np.hstack(
                    (virtual_obstacle, self.agent_position[:, i:i + 1] + np.array([[1.], [1]])))
        self.obstacle_with_other_agent = np.hstack((self.obstacle_total, virtual_obstacle))

        for i in range(self.num_agent):

            temp = np.argmin(np.linalg.norm(self.obstacle_total - self.agent_position[:, i:i + 1], axis=0))
            self.obstacle_closest[:, i:i + 1] = self.obstacle_total[:, temp:temp + 1]

            temp = np.argmin(np.linalg.norm(self.obstacle_with_other_agent - self.agent_position[:, i:i + 1], axis=0))
            self.obstacle_closest_with_other_agent[:, i:i + 1] = self.obstacle_with_other_agent[:, temp:temp + 1]

            temp1 = self.obstacle_closest_with_other_agent[:, i:i + 1] - self.agent_position[:, i:i + 1]
            angle1 = np.arccos(
                np.clip(
                    np.dot(np.ravel(temp1), np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                        temp1) / np.linalg.norm(self.agent_orientation[:, i:i + 1]), -1, 1)) / np.pi
            if np.cross(np.ravel(self.agent_orientation[:, i:i + 1]), np.ravel(temp1)) > 0:
                pass
            else:
                angle1 = -angle1

            temp2 = self.target_position - self.agent_position[:, i:i + 1]
            angle2 = np.arccos(
                np.clip(
                    np.dot(np.ravel(temp2), np.ravel(self.agent_orientation[:, i:i + 1])) / np.linalg.norm(
                        temp2) / np.linalg.norm(self.agent_orientation[:, i:i + 1]), -1, 1)) / np.pi
            if np.cross(np.ravel(self.agent_orientation[:, i:i + 1]), np.ravel(temp2)) > 0:
                pass
            else:
                angle2 = -angle2

            state = np.zeros((self.num_state,))
            state[:4] = np.array([np.linalg.norm(temp1) / 5000, angle1, np.linalg.norm(temp2) / 5000, angle2],
                                 dtype='float32')

            friends_position = np.delete(self.agent_position, i, axis=1)
            for j in range(self.num_agent - 1):
                friend_position = friends_position[:, j:j + 1]
                self_position = self.agent_position[:, i:i + 1]
                self_orientation = self.agent_orientation[:, i:i + 1]
                temp = friend_position - self_position
                distance = np.linalg.norm(temp)
                angle = np.arccos(
                    np.clip(
                        np.dot(np.ravel(temp), np.ravel(self_orientation)) / distance / np.linalg.norm(
                            self_orientation), -1, 1)) / np.pi
                if np.cross(np.ravel(self_orientation), np.ravel(temp)) > 0:
                    pass
                else:
                    angle = -angle

                if distance < self.r_perception:
                    state[5 + 2 * j] = np.linalg.norm(temp) / 5000
                    state[6 + 2 * j] = np.array(angle)
                else:
                    state[5 + 2 * j] = 2
                    state[6 + 2 * j] = 0
            if np.any(self.done == 1):
                state[4] = 1
            else:
                state[4] = 0
            self.state[:, i] = state
