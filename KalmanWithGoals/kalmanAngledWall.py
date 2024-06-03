import numpy as np
import pygame
import sys
import math
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class KalmanFilter:
    def __init__(self, dt, state_matrix, control_matrix, measurement_matrix, process_covariance, measurement_covariance, error_covariance):
        self.dt = dt
        self.A = state_matrix
        self.B = control_matrix
        self.H = measurement_matrix
        self.Q = process_covariance
        self.R = measurement_covariance
        self.P = error_covariance
        self.x = np.zeros((state_matrix.shape[0], 1))

    def predict(self, u=0, walls=None, Q=None, A=None):
        if Q is not None:
            self.Q = Q
        if A is not None:
            self.A = A

        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        if walls is not None:
            inside_wall, closest_side, wall = check_bounds(self.x, walls)
            if inside_wall:
                self.x = handle_wall_collision(self.x, wall, closest_side)

        return self.x

    def update(self, z, H_custom, R_custom):
        y = z - np.dot(H_custom, self.x)
        S = R_custom + np.dot(H_custom, np.dot(self.P, H_custom.T))
        K = np.dot(np.dot(self.P, H_custom.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(H_custom, self.P))

class AngledWall:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.angle = math.atan2(y2 - y1, x2 - x1)

    def draw(self, surface):
        pygame.draw.line(surface, (255, 255, 255), (self.x1, self.y1), (self.x2, self.y2), 2)

    def get_distance(self, x, y):
        # Distance from point (x, y) to the line segment (x1, y1) - (x2, y2)
        px, py = self.x2 - self.x1, self.y2 - self.y1
        norm = px*px + py*py
        u = ((x - self.x1) * px + (y - self.y1) * py) / float(norm)
        u = max(min(u, 1), 0)
        closest_x = self.x1 + u * px
        closest_y = self.y1 + u * py

        dx = closest_x - x
        dy = closest_y - y
        return math.sqrt(dx*dx + dy*dy)

    def collidepoint(self, x, y):
        distance = self.get_distance(x, y)
        if distance < 1:
            if min(self.x1, self.x2) <= x <= max(self.x1, self.x2) and min(self.y1, self.y2) <= y <= max(self.y1, self.y2):
                return True
        return False

class Wall:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 255, 255), self.rect)

def get_distance_to_wall(robot_position, robot_orientation, walls, screen):
    orientation_rad = math.radians(robot_orientation.item())
    start_x, start_y = int(robot_position[0, 0]), int(robot_position[1, 0])
    start_point = (start_x, start_y)
    max_sensor_distance = 1000
    end_x = start_x + max_sensor_distance * math.cos(orientation_rad)
    end_y = start_y + max_sensor_distance * math.sin(orientation_rad)
    end_point = (int(end_x), int(end_y))

    pygame.draw.line(screen, (255, 0, 0), start_point, end_point, 1)

    min_distance = max_sensor_distance
    wall_collide = None
    for wall in walls:
        if isinstance(wall, AngledWall):
            intersect_point = wall_intersection_point(wall, start_point, (end_x, end_y))
            if intersect_point:
                distance = math.hypot(intersect_point[0] - start_x, intersect_point[1] - start_y)
                if distance < min_distance:
                    min_distance = distance
                    wall_collide = wall
        else:
            if wall.rect.clipline(start_point, end_point):
                points = wall.rect.clipline(start_point, end_point)
                for point in points:
                    distance = math.hypot(point[0] - start_x, point[1] - start_y)
                    if distance < min_distance:
                        min_distance = distance
                        wall_collide = wall

    return min_distance, wall_collide

def wall_intersection_point(wall, start_point, end_point):
    """ Returns the intersection point of the sensor line with the wall, or None if they don't intersect """
    x1, y1, x2, y2 = wall.x1, wall.y1, wall.x2, wall.y2
    x3, y3, x4, y4 = start_point[0], start_point[1], end_point[0], end_point[1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

    if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2):
        return (px, py)
    return None

def check_collision(robot_rect, walls):
    for wall in walls:
        if isinstance(wall, AngledWall):
            if wall.collidepoint(robot_rect.centerx, robot_rect.centery):
                return True
        else:
            if robot_rect.colliderect(wall.rect):
                return True
    return False

def handle_wall_collision(state, wall, closest_side):
    if isinstance(wall, AngledWall):
        distance = wall.get_distance(state[0, 0], state[1, 0])
        px, py = wall.x2 - wall.x1, wall.y2 - wall.y1
        norm = px * px + py * py
        u = ((state[0, 0] - wall.x1) * px + (state[1, 0] - wall.y1) * py) / float(norm)
        u = max(min(u, 1), 0)
        closest_x = wall.x1 + u * px
        closest_y = wall.y1 + u * py

        adjust_x = closest_x + (state[0, 0] - closest_x) / distance
        adjust_y = closest_y + (state[1, 0] - closest_y) / distance
        state[0, 0] = adjust_x
        state[1, 0] = adjust_y
    else:
        if closest_side == 'left':
            state[0] = wall.rect.left - 1
        elif closest_side == 'right':
            state[0] = wall.rect.right + 1
        elif closest_side == 'top':
            state[1] = wall.rect.top - 1
        elif closest_side == 'bottom':
            state[1] = wall.rect.bottom + 1

    return state

def move_robot(true_state, control_inputs, walls, dt, drift_x, drift_y, wind_x, wind_y, angle_noise):
    linear_acceleration, angular_acceleration = control_inputs[0, 0], control_inputs[1, 0]
    temp_true = true_state.copy()
    
    true_state[3] = linear_acceleration * (np.random.normal(1, .2))
    true_state[5] = angular_acceleration * (np.random.normal(1, .2))
    
    orientation_rad = math.radians(true_state[2, 0].item())
    drift_x = math.cos(drift_angle + orientation_rad) * drift_velocity
    drift_y = math.sin(drift_angle + orientation_rad) * drift_velocity
    
    true_state[0] += true_state[3] * dt * math.cos(orientation_rad + angle_noise) + (drift_x + wind_x) * dt
    true_state[1] += true_state[3] * dt * math.sin(orientation_rad + angle_noise) + (drift_y + wind_y) * dt
    true_state[2] += math.degrees(true_state[5] * dt) + np.random.normal(0, 2) * dt

    robot_rect = pygame.Rect(int(true_state[0, 0]), int(true_state[1, 0]), 1, 1)

    if check_collision(robot_rect, walls):
        true_state[0] = temp_true[0]
        true_state[1] = temp_true[1]
        true_state[3] = 0

    return true_state

def draw_robot(surface, position, orientation, color):
    robot_size = 20
    center = (int(position[0, 0]), int(position[1, 0]))
    angle = math.radians(orientation.item())
    points = [
        (center[0] + robot_size * math.cos(angle), center[1] + robot_size * math.sin(angle)),
        (center[0] + robot_size * math.cos(angle + 2.2), center[1] + robot_size * math.sin(angle + 2.2)),
        (center[0] + robot_size * math.cos(angle - 2.2), center[1] + robot_size * math.sin(angle - 2.2))
    ]
    pygame.draw.polygon(surface, color, points)

def draw_ellipse(surface, color, position, covariance, scale=2):
    position = tuple(map(int, position.flatten()))
    pos_covariance = covariance[:2, :2]
    eigenvalues, eigenvectors = np.linalg.eig(pos_covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    width, height = np.sqrt(eigenvalues) * scale * 2
    width, height = int(width), int(height)
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    max_dimension = max(width, height)
    ellipse_surface = pygame.Surface((max_dimension * 2, max_dimension * 2), pygame.SRCALPHA)
    ellipse_surface.fill((0, 0, 0, 0))
    ellipse_rect = pygame.Rect(max_dimension - width // 2, max_dimension - height // 2, width, height)
    pygame.draw.ellipse(ellipse_surface, color, ellipse_rect, 1)
    rotated_surface = pygame.transform.rotate(ellipse_surface, np.degrees(-angle))

    try:
        rotated_rect = rotated_surface.get_rect(center=position)
    except TypeError as e:
        logging.error(f"Error in rotated_rect assignment: {e}")
        logging.error(f"Position value causing error: {position}")
        rotated_rect = rotated_surface.get_rect(center=position)

    surface.blit(rotated_surface, rotated_rect.topleft)

def render_text(screen, text, position, font_size=30, color=(255, 255, 255)):
    font = pygame.font.SysFont(None, font_size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

def rotate_covariance_matrix(theta, sigma_radial, sigma_perpendicular):
    R_theta = np.array([[np.cos(theta), -np.sin(theta)], 
                        [np.sin(theta), np.cos(theta)]])
    R_unrotated = np.array([[sigma_radial**2, 0], 
                            [0, sigma_perpendicular**2]])
    R_rotated = R_theta @ R_unrotated @ R_theta.T
    np.linalg.svd(R_rotated)
    return R_rotated

def kalRobotCoord(dt, Q, P, A):
    B = np.array([[dt, 0, 0],
                  [0, dt, 0],
                  [0, 0, dt],
                  [0, 0, 0],  
                  [0, 0, 0],  
                  [0, 0, 0],  
                  [0, 0, 0],  
                  [0, 0, 0]])
    
    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 1, 0, 0, 0, 0, 0, 0]])
    
    kf = KalmanFilter(dt, A, B, H, Q, np.eye(3) * 10, P)
    return kf

def position_measurement_available():
    return np.random.random() > 0.95

def get_position_measurement(true_state):
    position_noise = np.random.normal(0, 10, size=(2, 1))
    measured_position = true_state[:2] + position_noise
    return measured_position

def angle_measurement_available():
    return np.random.random() > 0.1

def get_angle_measurement(true_state):
    angle_measurement = np.array([[true_state[2, 0].item()]]) + np.random.normal(0, .1)
    return angle_measurement

# Simulation parameters
dt = 0.2
initial_covariance_value = 20
Q = np.eye(8) * 0.001
Q[3][3] = 0.001
Q[4][4] = 0.001
Q[6][6] = 0.0001
Q[7][7] = 0.0001
P = np.eye(8) * initial_covariance_value
A = np.array([
    [1, 0, 0, dt, 0, 0, 0, 0],
    [0, 1, 0, 0, dt, 0, 0, 0],
    [0, 0, 1, 0, 0, dt, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

# Wind and drift parameters
angle_noise = 0
wind_angle = -4 * np.pi / 3
wind_velocity = 1
drift_angle = np.pi / 4
drift_velocity = 0

drift_x = math.cos(drift_angle) * drift_velocity
drift_y = math.sin(drift_angle) * drift_velocity
wind_x = math.cos(wind_angle) * wind_velocity
wind_y = math.sin(wind_angle) * wind_velocity

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Initialize robot state
true_state = np.array([[400.0], [300.0], [90.0], [0.0], [0.0], [0.0]])

# Define walls
walls = [
    Wall(100, 100, 600, 20), 
    Wall(100, 480, 600, 20),
    Wall(100, 120, 20, 360), 
    Wall(680, 120, 20, 360)
]

# Calculate center points of the top and right walls
top_wall_center_x = 100 + 600 // 2
top_wall_center_y = 100
right_wall_center_x = 680
right_wall_center_y = 120 + 360 // 2

# Add the angled wall
angled_wall = AngledWall(top_wall_center_x, top_wall_center_y, right_wall_center_x, right_wall_center_y)
walls.append(angled_wall)

# Initialize Kalman Filter
kf = kalRobotCoord(dt, Q, P, A)
kf.x = np.vstack((true_state, np.zeros((2, 1))))  # Initialize the Kalman Filter state to the true state with zeros for the additional state variables

def check_bounds(state, walls):
    x, y = state[0, 0], state[1, 0]
    for wall in walls:
        if isinstance(wall, AngledWall):
            if wall.collidepoint(x, y):
                distances = {
                    'left': abs(wall.x1 - x),
                    'right': abs(wall.x2 - x),
                    'top': abs(wall.y1 - y),
                    'bottom': abs(wall.y2 - y)
                }
                closest_side = min(distances, key=distances.get)
                return True, closest_side, wall
        else:
            if wall.rect.collidepoint(x, y):
                distances = {
                    'left': abs(wall.rect.left - x),
                    'right': abs(wall.rect.right - x),
                    'top': abs(wall.rect.top - y),
                    'bottom': abs(wall.rect.bottom - y)
                }
                closest_side = min(distances, key=distances.get)
                return True, closest_side, wall
    return False, None, None

def rotate_covariance(covariance, angle, indices=(0, 1)):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    rotated_covariance = covariance.copy()
    submatrix = covariance[np.ix_(indices, indices)]
    rotated_submatrix = rotation_matrix @ submatrix @ rotation_matrix.T
    rotated_covariance[np.ix_(indices, indices)] = rotated_submatrix
    return rotated_covariance

def inverse_rotate_covariance(covariance, angle, indices=(0, 1)):
    inverse_rotation_matrix = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    rotated_covariance = covariance.copy()
    submatrix = covariance[np.ix_(indices, indices)]
    rotated_submatrix = inverse_rotation_matrix @ submatrix @ inverse_rotation_matrix.T
    rotated_covariance[np.ix_(indices, indices)] = rotated_submatrix
    return rotated_covariance

def translate_state(state, translation_vector):
    translated_state = state.copy()
    translated_state[:2] -= translation_vector.reshape(2, 1)
    return translated_state

def rotate_state(state, angle):
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    position = state[:2]
    rotated_position = np.dot(rotation_matrix, position)
    rotated_state = state.copy()
    rotated_state[:2] = rotated_position
    return rotated_state

def inverse_rotate_state(state, angle):
    inverse_rotation_matrix = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    position = state[:2]
    rotated_position = np.dot(inverse_rotation_matrix, position)
    rotated_state = state.copy()
    rotated_state[:2] = rotated_position
    return rotated_state

def inverse_translate_state(state, translation_vector):
    translated_state = state.copy()
    translated_state[:2] += translation_vector.reshape(2, 1)
    return translated_state





time = 0
turn_time = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    distance_to_wall, wall_collide = get_distance_to_wall(true_state[:2], true_state[2], walls, screen)
    if distance_to_wall < 80 and turn_time <= 0:
        turn_time = np.random.random() * 2 + 2
        total_turn_time = turn_time

    linear_velocity = 20
    angular_velocity = 0
    if turn_time > 0:
        linear_velocity = 0
        angular_velocity = 0.3
        turn_time -= dt

    control_inputs = np.array([[linear_velocity], [angular_velocity]])
    true_state = move_robot(true_state, control_inputs, walls, dt, drift_x, drift_y, wind_x, wind_y, angle_noise)
    
    inside_wall, closest_side, wall = check_bounds(true_state, walls)
    if inside_wall:
        true_state = handle_wall_collision(true_state, wall, closest_side)

    estx = kf.x[0, 0]
    esty = kf.x[1, 0]
    orientation_rad2 = math.radians(kf.x[2, 0].item())
    orientation_rad = math.atan2(math.sin(orientation_rad2), math.cos(orientation_rad2))

    predict_control = np.array([[linear_velocity * math.cos(orientation_rad2)], [linear_velocity * math.sin(orientation_rad2)], [angular_velocity]])
    rotate_estimation = np.array([[np.cos(orientation_rad), -np.sin(orientation_rad)], [np.sin(orientation_rad), np.cos(orientation_rad)]])
    Q_rotate = rotate_covariance_matrix(orientation_rad, 1, 0.01)
    Q[0:2, 0:2] = Q_rotate
    A[0:2, 6:8] = rotate_estimation
    kf.predict(predict_control, walls=walls, Q=Q, A=A)

    inside_wall, closest_side, wall = check_bounds(kf.x, walls)
    if inside_wall:
        kf.x = handle_wall_collision(kf.x, wall, closest_side)

    if distance_to_wall < 125 and angular_velocity == 0:
        if wall_collide is not None:
            if isinstance(wall_collide, AngledWall):
                # Calculate translation and rotation
                translation_vector = np.array([wall_collide.x1, wall_collide.y1])
                rotation_angle = -wall_collide.angle

                # Translate and rotate the state and covariance
                translated_state = translate_state(kf.x, translation_vector)
                rotated_state = rotate_state(translated_state, rotation_angle)
                rotated_covariance = rotate_covariance(kf.P, rotation_angle, indices=(0, 1))

                # Perform measurement update in rotated frame
                kf.x = rotated_state  # Update the state before the update
                kf.P = rotated_covariance  # Update the covariance before the update
                H_tof = np.array([[1, 0, 0, 0, 0, 0, 0, 0]])
                oriented_dist = wall_collide.get_distance(estx, esty)
                kf.update(oriented_dist, H_tof, np.array([[10000]]))

                # Rotate and translate back the state and covariance
                rotated_back_state = inverse_rotate_state(kf.x, rotation_angle)
                kf.x = inverse_translate_state(rotated_back_state, translation_vector)
                kf.P = inverse_rotate_covariance(kf.P, rotation_angle, indices=(0, 1))
            else:
                if wall_collide.rect.width > wall_collide.rect.height:
                    H_tof = np.array([[0, 1, 0, 0, 0, 0, 0, 0]])
                    oriented_dist = wall_collide.rect.y - np.sin(orientation_rad) * np.random.normal(distance_to_wall, distance_to_wall * 0.2)
                else:
                    H_tof = np.array([[1, 0, 0, 0, 0, 0, 0, 0]])
                    oriented_dist = wall_collide.rect.x - np.cos(orientation_rad) * np.random.normal(distance_to_wall, distance_to_wall * 0.2)
                kf.update(oriented_dist, H_tof, np.array([[10000]]))



    if angle_measurement_available():
        angle_measurement = get_angle_measurement(true_state)
        kf.update(angle_measurement, np.array([[0, 0, 1, 0, 0, 0, 0, 0]]), np.array([[0.1]]))

    screen.fill((0, 0, 0))
    for wall in walls:
        wall.draw(screen)

    draw_robot(screen, true_state[:2], true_state[2], (0, 0, 255))
    draw_robot(screen, kf.x[:2], kf.x[2], (255, 0, 0))
    draw_ellipse(screen, (0, 255, 0), kf.x[:2], kf.P)

    orientation_rad = math.radians(true_state[2, 0].item())
    sensor_start_x, sensor_start_y = int(true_state[0, 0]), int(true_state[1, 0])
    sensor_end_x = sensor_start_x + min(distance_to_wall, 125) * math.cos(orientation_rad)
    sensor_end_y = sensor_start_y + min(distance_to_wall, 125) * math.sin(orientation_rad)
    pygame.draw.line(screen, (255, 0, 0), (sensor_start_x, sensor_start_y), (int(sensor_end_x), int(sensor_end_y)), 1)

    true_state_text = f"True State: x={true_state[0, 0].item():.2f}, y={true_state[1, 0].item():.2f}, angle={math.degrees(math.atan2(math.cos(math.radians(true_state[2, 0].item())), math.sin(math.radians(true_state[2, 0].item())))):.2f}, vx={wind_x:.2f}, vy={wind_y:.2f}"
    estimated_state_text = f"Estimated: x={kf.x[0, 0].item():.2f}, y={kf.x[1, 0].item():.2f}, angle={math.degrees(math.atan2(math.cos(math.radians(kf.x[2, 0].item())), math.sin(math.radians(kf.x[2, 0].item())))):.2f}, vx={kf.x[3, 0].item():.2f}, vy={kf.x[4, 0].item():.2f}"
    control_text = f"Estimated drift: x={(kf.x[6, 0].item() / dt):.2f}, y={(kf.x[7, 0].item() / dt):.2f}, True drift: x={(drift_x):.2f}, y={(drift_y):.2f}"
    render_text(screen, true_state_text, (10, 10))
    render_text(screen, estimated_state_text, (10, 40))
    render_text(screen, control_text, (10, 60))

    pygame.display.flip()
    pygame.time.delay(20)
    clock.tick(50)
    time += dt
