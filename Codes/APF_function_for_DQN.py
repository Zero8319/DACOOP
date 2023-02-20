import numpy as np


def attract(self_position, target_position):
    F = (target_position - self_position) / np.linalg.norm(target_position - self_position)
    return F


def repulse(self_position, obstacle_closest, influence_range, scale_repulse):
    F = scale_repulse * (1 / (np.linalg.norm(self_position - obstacle_closest) - 100) - 1 / influence_range) / (
            np.linalg.norm(self_position - obstacle_closest) - 100) ** 2 * (
                self_position - obstacle_closest) / np.linalg.norm(self_position - obstacle_closest)

    if np.linalg.norm(self_position - obstacle_closest) < influence_range:
        return F
    else:
        return np.array([[0], [0]])


def individual(self_position, friend_position, individual_balance, r_perception):
    F = np.zeros((2, 0))
    for i in range(friend_position.shape[1]):
        temp = (friend_position[:, i:i + 1] - self_position) / np.linalg.norm(
            friend_position[:, i:i + 1] - self_position) * (
                       0.5 - individual_balance / (np.linalg.norm(friend_position[:, i:i + 1] - self_position) - 200))
        if np.linalg.norm(friend_position[:, i:i + 1] - self_position) < r_perception:
            F = np.hstack((F, temp))
    if F.size == 0:
        F = np.zeros((2, 1))
    return np.mean(F, axis=1, keepdims=True)


def generate_boundary(point1, point2, point3, point4, step):
    temp1 = np.ravel(np.arange(point1[0], point2[0] + 1, step))
    temp2 = np.ravel(np.arange(point2[1], point3[1] + 1, step))
    boundary12 = np.vstack((temp1, np.ones_like(temp1) * point1[1]))
    boundary23 = np.vstack((np.ones_like(temp2) * point2[0], temp2))
    boundary34 = np.vstack((np.flipud(temp1), np.ones_like(temp1) * point3[1]))
    boundary41 = np.vstack((np.ones_like(temp2) * point4[0], np.flipud(temp2)))

    boundary = np.hstack((boundary12, boundary23, boundary34, boundary41))
    return boundary


def generate_obstacle(center):
    boundary12 = np.vstack(
        (np.ravel(np.linspace(center[0] - 50, center[0] + 50, 50)),
         np.ravel(np.linspace(center[1] - 50, center[1] - 50, 50))))
    boundary23 = np.vstack(
        (np.ravel(np.linspace(center[0] + 50, center[0] + 50, 50)),
         np.ravel(np.linspace(center[1] - 50, center[1] + 50, 50))))
    boundary34 = np.vstack(
        (np.ravel(np.linspace(center[0] - 50, center[0] + 50, 50)),
         np.ravel(np.linspace(center[1] + 50, center[1] + 50, 50))))
    boundary41 = np.vstack(
        (np.ravel(np.linspace(center[0] - 50, center[0] - 50, 50)),
         np.ravel(np.linspace(center[1] + 50, center[1] - 50, 50))))

    boundary = np.hstack((boundary12, boundary23, boundary34, boundary41))
    return boundary


def wall_follow(self_orientation, F_repulse, F_individual):
    rotate_matrix = np.array([[0, -1], [1, 0]])
    rotate_vector1 = np.matmul(rotate_matrix, F_repulse)
    rotate_vector2 = -1 * rotate_vector1
    temp1 = np.linalg.norm(rotate_vector1 - self_orientation)
    temp2 = np.linalg.norm(rotate_vector2 - self_orientation)
    if np.linalg.norm(F_individual) < 1:
        if temp1 > temp2:
            return rotate_vector2
        else:
            return rotate_vector1
    else:
        if np.dot(np.ravel(rotate_vector1), np.ravel(F_individual)) > 0:
            return rotate_vector1
        else:
            return rotate_vector2


def wall_follow_for_escaper(F_repulse, target_orientation, distance_from_nearest_agent, F_escape,
                            distance_from_nearest_obstacle):
    rotate_matrix = np.array([[0, -1], [1, 0]])
    rotate_vector1 = np.matmul(rotate_matrix, F_repulse)
    rotate_vector2 = -1 * rotate_vector1

    if np.dot(np.ravel(target_orientation), np.ravel(rotate_vector1)) > 0:
        final = rotate_vector1
        if distance_from_nearest_agent < 400 and np.dot(np.ravel(F_escape), np.ravel(rotate_vector2)) * 1.5 > np.dot(
                np.ravel(F_escape), -np.ravel(F_repulse)):
            final = rotate_vector2
    else:
        final = rotate_vector2
        if distance_from_nearest_agent < 400 and np.dot(np.ravel(F_escape), np.ravel(rotate_vector1)) * 1.5 > np.dot(
                np.ravel(F_escape), -np.ravel(F_repulse)):
            final = rotate_vector1
    if distance_from_nearest_obstacle < 50:
        final = final + F_repulse
    return final


# def wall_follow_for_escaper(F_repulse, target_orientation, distance_from_nearest_agent, F_escape):
#     rotate_matrix = np.array([[0, -1], [1, 0]])
#     rotate_vector1 = np.matmul(rotate_matrix, F_repulse)
#     rotate_vector2 = -1 * rotate_vector1
#
#     if np.dot(np.ravel(target_orientation), np.ravel(rotate_vector1)) > 0:
#         final = rotate_vector1
#     else:
#         final = rotate_vector2
#
#     if distance_from_nearest_agent < 150:
#         if np.dot(np.ravel(F_escape), np.ravel(rotate_vector1)) > 0:
#             final = rotate_vector1
#         else:
#             final = rotate_vector2
#     return final


def APF_decision(self_position, friend_position, target_position, obstacle_closest, scale_repulse, individual_balance,
                 r_perception):
    influence_range = 800
    F_attract = attract(self_position, target_position)
    F_repulse = repulse(self_position, obstacle_closest, influence_range, scale_repulse)
    F_individual = individual(self_position, friend_position, individual_balance, r_perception)
    F = F_attract + F_repulse + F_individual
    return F_attract, F_repulse, F_individual, F


def total_decision(agent_position, agent_orientation, obstacle_closest, target_position, scale_repulse,
                   individual_balance, r_perception):
    F = np.zeros((2, 0))
    wall_following = np.zeros((1, 0))
    for i in range(scale_repulse.size):
        self_position = agent_position[:, i:i + 1]
        friend_position = np.delete(agent_position, i, axis=1)
        self_orientation = agent_orientation[:, i:i + 1]
        F_attract, F_repulse, F_individual, F_temp = APF_decision(self_position, friend_position, target_position,
                                                                  obstacle_closest[:, i:i + 1], scale_repulse[0, i],
                                                                  individual_balance[0, i], r_perception)
        vector1 = np.ravel(F_attract + F_repulse)
        vector2 = np.ravel(F_attract)

        if np.dot(vector1, vector2) < 0:
            F_temp = wall_follow(self_orientation, F_repulse, F_individual)
            wall_following = np.hstack((wall_following, np.array(True).reshape(1, 1)))
        else:
            wall_following = np.hstack((wall_following, np.array(False).reshape(1, 1)))
        F_temp = F_temp / np.linalg.norm(F_temp)
        F = np.hstack((F, F_temp))
    return F, wall_following


def is_target_in_obstacle(obstacle_center, target):
    for i in range(obstacle_center.shape[1]):
        xmin = obstacle_center[0, i] - 50
        xmax = obstacle_center[0, i] + 50
        ymin = obstacle_center[1, i] - 50
        ymax = obstacle_center[1, i] + 50
    if all((target[0] < xmax, target[0] > xmin, target[1] > ymin, target[1] < ymax)):
        return True
    else:
        return False
