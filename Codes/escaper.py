import numpy as np
import APF_function_for_DQN

v_max = 400
v_agent = 300
rcd = 200  # 捕捉半径
p_thres = 0.2
sensitivity_range = 2000
r_zigzag = 300
influence_range = 800
scale_repulse = 1e7
slip_range = 500


def escaper(agent_position, target_position, target_orientation, obstacle_total, num_agent, zigzag_count, zigzag_last,
            last_e, slip_flag):
    distance_from_agent = np.linalg.norm(target_position - agent_position, axis=0, keepdims=True)
    distance_from_obstacle = np.linalg.norm(target_position - obstacle_total, axis=0, keepdims=True)
    agent_in_sensitivity_range = distance_from_agent < sensitivity_range
    agent_in_slip_range = distance_from_agent < slip_range
    ##########直接逃逸或zigzag#############
    distance_from_nearest_agent = np.min(distance_from_agent)
    distance_from_nearest_obstacle = np.min(distance_from_obstacle)

    if distance_from_nearest_agent > sensitivity_range:
        p_panic = 0
    else:
        p_panic = 1 / (np.exp(1) - 1) * (np.exp(-distance_from_nearest_agent / sensitivity_range + 1) - 1)

    if (p_panic < p_thres) and (distance_from_nearest_obstacle > r_zigzag):
        zigzag_flag = 1
        if zigzag_count > np.random.randint(10, 15):
            zigzag_count = 0
        if zigzag_count == 0:
            while True:
                temp = np.random.random() * 2 * np.pi - np.pi
                F_escape = np.array([[np.cos(temp)],
                                     [np.sin(temp)]])
                if np.dot(np.ravel(F_escape), np.ravel(target_orientation)) > 0:
                    zigzag_last = F_escape
                    break
            zigzag_count += 1
        else:
            F_escape = zigzag_last
            zigzag_count += 1
    else:
        zigzag_flag = 0
        zigzag_count = 0
        F_escape = np.full((2, num_agent), np.nan)

        for i in range(num_agent):
            if np.ravel(agent_in_sensitivity_range)[i]:
                F_escape[:, i:i + 1] = (target_position - agent_position)[:, i:i + 1] / np.linalg.norm(
                    distance_from_agent[:, i]) ** 2

        if np.all(np.isnan(F_escape)):
            F_escape = np.zeros((2, 1))
        else:
            F_escape = np.nanmean(F_escape, axis=1, keepdims=True) / np.linalg.norm(np.nanmean(F_escape, axis=1))
    ###########排斥力############
    obstacle_sort_index = np.argsort(distance_from_obstacle)
    obstacle_closest = obstacle_total[:, np.ravel(obstacle_sort_index)[0:10]]

    F_repulse = np.zeros((2, 10))
    for i in range(10):
        F_repulse[:, i:i + 1] = APF_function_for_DQN.repulse(target_position, obstacle_closest[:, i:i + 1],
                                                             influence_range,
                                                             scale_repulse)
    F_repulse = np.mean(F_repulse, axis=1, keepdims=True)
    #############total force############
    vector1 = np.ravel(F_escape + F_repulse)
    vector2 = np.ravel(F_escape)

    if np.dot(vector1, vector2) < 0:
        F_temp = APF_function_for_DQN.wall_follow_for_escaper(F_repulse, target_orientation,
                                                              distance_from_nearest_agent, F_escape,
                                                              distance_from_nearest_obstacle)
        F_total = F_temp / np.linalg.norm(np.ravel(F_temp))
        wall_following = 1
    else:
        F_temp = F_repulse + F_escape
        F_total = F_temp / np.linalg.norm(np.ravel(F_temp))
        wall_following = 0

    ############额外规则############
    # if np.sum(agent_in_sensitivity_range) > 1 and distance_from_nearest_obstacle < 300:
    if np.sum(agent_in_slip_range) > 1:
        agent_closest_index = np.argsort(distance_from_agent)
        agent_closest1 = agent_position[:, agent_closest_index[0, 0]].reshape(2, -1)
        agent_closest2 = agent_position[:, agent_closest_index[0, 1]].reshape(2, -1)
        e_temp = (agent_closest1 - target_position) + (agent_closest2 - target_position)
        e = e_temp / np.linalg.norm(np.ravel(e_temp))
        rp1 = e * np.dot(np.ravel(e), np.ravel(agent_closest1 - target_position))
        rp2 = e * np.dot(np.ravel(e), np.ravel(agent_closest2 - target_position))
        taue1 = np.linalg.norm(np.ravel(rp1)) / v_max
        taue2 = np.linalg.norm(np.ravel(rp2)) / v_max
        tauc1 = (np.linalg.norm(np.ravel(rp1 - agent_closest1 + target_position)) - rcd) / v_agent
        tauc2 = (np.linalg.norm(np.ravel(rp2 - agent_closest2 + target_position)) - rcd) / v_agent
        if slip_flag == 0:
            if taue1 < tauc1 and taue2 < tauc2 and distance_from_nearest_obstacle < 300 and np.dot(np.ravel(F_repulse),
                                                                                                   np.ravel(e)) >= 0:
                F_total = e
                last_e = e
                slip_flag = 1
            else:
                slip_flag = 0
        else:
            if taue1 < tauc1 and taue2 < tauc2 and distance_from_nearest_obstacle > 100 and np.dot(np.ravel(e),
                                                                                                   np.ravel(
                                                                                                       last_e)) >= 0:
                F_total = e
                slip_flag = 1
            elif taue1 < tauc1 and taue2 < tauc2 and distance_from_nearest_obstacle > 100 and np.dot(np.ravel(e),
                                                                                                     np.ravel(
                                                                                                         last_e)) < 0:
                F_total = -e
                slip_flag = 1
            else:
                slip_flag = 0
        # if taue1 < tauc1 and taue2 < tauc2 and (
    #             np.dot(np.ravel(F_repulse),
    #                    np.ravel(e)) >= 0 or distance_from_nearest_obstacle > 100):
    #         if slip_flag == 0:
    #             F_total = e
    #             last_e = e
    #         else:
    #             if np.dot(np.ravel(e), np.ravel(last_e)) >= 0:
    #                 F_total = e
    #             else:
    #                 F_total = -e
    #         slip_flag = 1
    #     else:
    #         slip_flag = 0
    # else:
    #     slip_flag = 0

    F_total = F_total * v_max
    return F_total, zigzag_count, zigzag_last, zigzag_flag, wall_following, slip_flag, distance_from_nearest_obstacle, last_e
