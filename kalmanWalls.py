import numpy as np
import pygame
import sys
import math

class KalmanFilter:
    def __init__(self, dt, state_matrix, control_matrix, measurement_matrix, process_covariance, measurement_covariance, error_covariance):
        self.dt = dt
        self.A = state_matrix
        self.B = control_matrix
        self.H = measurement_matrix  # Initial measurement matrix (can be either H_position or H_angle)
        self.Q = process_covariance
        self.R = measurement_covariance
        self.P = error_covariance
        self.x = np.zeros((state_matrix.shape[0], 1))

    def set_measurement_matrix(self, H, R):
        self.H = H
        self.R = R


    def predict(self, u=0):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))


class Wall:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)

    def draw(self, surface):
        pygame.draw.rect(surface, (255, 255, 255), self.rect)  # Draw wall in white

def get_distance_to_wall(robot_position, robot_orientation, walls, screen):
    # Convert orientation to radians
    orientation_rad = math.radians(robot_orientation)

    # Starting point of the sensor ray
    start_x, start_y = int(robot_position[0]), int(robot_position[1])
    start_point = (start_x, start_y)

    # Arbitrary large number
    max_sensor_distance = 1000

    # Calculate the end point of the sensor ray
    end_x = start_x + max_sensor_distance * math.cos(orientation_rad)
    end_y = start_y + max_sensor_distance * math.sin(orientation_rad)
    end_point = (int(end_x), int(end_y))

    # Draw the sensor line (for debugging)
    pygame.draw.line(screen, (255, 0, 0), start_point, end_point, 1)

    # Check for intersection with each wall
    min_distance = max_sensor_distance
    for wall in walls:
        # Check if line intersects with wall rectangle
        if wall.rect.clipline(start_point, end_point):
            # Pygame's clipline returns a list of points
            points = wall.rect.clipline(start_point, end_point)
            if points:
                distance = math.hypot(points[0][0] - start_x, points[0][1] - start_y)
                min_distance = min(min_distance, distance)

    return min_distance


# Define walls in your game world
walls = [Wall(100, 100, 600, 20), Wall(100, 480, 600, 20),
         Wall(100, 120, 20, 360), Wall(680, 120, 20, 360)]

def check_collision(robot_rect, walls):
    for wall in walls:
        if robot_rect.colliderect(wall.rect):
            return True
    return False

def draw_robot(surface, position, orientation, color):
    """ Draw the robot as a triangle representing its orientation. """
    robot_size = 20
    center = (int(position[0]), int(position[1]))
    angle = math.radians(orientation)
    points = [
        (center[0] + robot_size * math.cos(angle), center[1] + robot_size * math.sin(angle)),
        (center[0] + robot_size * math.cos(angle + 2.2), center[1] + robot_size * math.sin(angle + 2.2)),
        (center[0] + robot_size * math.cos(angle - 2.2), center[1] + robot_size * math.sin(angle - 2.2))
    ]
    pygame.draw.polygon(surface, color, points)


def draw_ellipse(surface, color, position, covariance):
    """ Draw an ellipse representing the uncertainty in position. """
    # Ensure only the position covariance is used (top-left 2x2 submatrix)
    pos_covariance = covariance[:2, :2]
    width, height = np.sqrt(np.diag(pos_covariance)) * 2  # Scale factor for visualization
    width, height = int(width), int(height)  # Convert to integers
    ellipse_rect = pygame.Rect(int(position[0] - width / 2), int(position[1] - height / 2), width, height)
    pygame.draw.ellipse(surface, color, ellipse_rect, 1)

def render_text(screen, text, position, font_size=30, color=(255, 255, 255)):
    font = pygame.font.SysFont(None, font_size)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)



# Pygame Initialization
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Initialize robot state with expanded state vector
true_state = np.array([[400.0], [300.0], [0.0], [0.0], [0.0], [0.0]])  # Include velocity components


# Kalman Filter Setup with expanded state vector
dt = 0.1  # Time step
drag_coefficient = 1  # This should be between 0 and 1, closer to 1
A = np.array([
    [1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0, 0],         # x_position updated with x_velocity and x_acceleration
    [0, 1, 0, 0, dt, 0, 0, 0.5 * dt**2, 0],         # y_position updated with y_velocity and y_acceleration
    [0, 0, 1, 0, 0, dt, 0, 0, 0.5 * dt**2],         # orientation updated with angular_velocity and angular_acceleration
    [0, 0, 0, 1, 0, 0, dt, 0, 0],                   # x_velocity updated with x_acceleration
    [0, 0, 0, 0, 1, 0, 0, dt, 0],                   # y_velocity updated with y_acceleration
    [0, 0, 0, 0, 0, 1, 0, 0, dt],                   # angular_velocity updated with angular_acceleration
    [0, 0, 0, 0, 0, 0, 1, 0, 0],                    # x_acceleration (assumed constant for this timestep)
    [0, 0, 0, 0, 0, 0, 0, 1, 0],                    # y_acceleration (assumed constant for this timestep)
    [0, 0, 0, 0, 0, 0, 0, 0, 1]])                   # angular_acceleration (assumed constant for this timestep)

B = np.array([[0, 0],
              [0, 0],
              [0, 0],
              [0, 0],  # Control input 1 affects x_velocity
              [0, 0],  # Control input 2 affects y_velocity
              [0, 0],
              [0, 0],
              [0, 0],
              [0, 0]])  # Assuming no direct control input for angular velocity


# Measurement matrix temp
H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],  # Measure x_position
                       [0, 1, 0, 0, 0, 0, 0, 0, 0]]) # Measure y_position
# Measurement matrix for position
# Measurement matrix for position (assuming measurements are x and y position)
H_position = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],  # Measure x_position
                       [0, 1, 0, 0, 0, 0, 0, 0, 0]]) # Measure y_position

# Measurement matrix for angle (assuming measurement is orientation)
H_angle = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0]])    # Measure orientation
position_measurement_noise_variance = 2500
angle_measurement_noise_variance = 1
R_position = np.eye(2) * position_measurement_noise_variance
R_angle = np.array([[angle_measurement_noise_variance]])

initial_covariance_value =10000

Q = np.eye(9) * .01
R = np.eye(3) * .02
P = np.eye(9) * initial_covariance_value  # initial_covariance_value is a tuning parameter

kf = KalmanFilter(dt, A, B, H, Q, R, P)

def position_measurement_available(true_state):
    if np.random.random() > .7:
        return True
    return False

def get_position_measurement(true_state):
    # Simulate Measurement with Some Noise for x and y position
    position_noise = np.random.normal(0, 2, size=(2, 1))  # Noise with standard deviation of 10
    measured_position = true_state[:2] + position_noise


    return measured_position

def angle_measurement_available(true_state):
    if np.random.random() > .1:
        return True
    return False

def get_angle_measurement(true_state):
    angle_measurement = np.array([[true_state[2, 0]]])
    return angle_measurement

# Main Loop
time = 0
turn_time = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()



    # Simulate Control Inputs (adapt as needed)
    # if time %100 < 25:
    #     linear_velocity = 20
    #     angular_velocity = 0
    # elif time %100 < 50:
    #     linear_velocity = 0
    #     angular_velocity = 5
    # elif time %100 < 75:
    #     linear_velocity = 20
    #     angular_velocity = 0
    # elif time %100 < 100:
    #     linear_velocity = 0
    #     angular_velocity = 5
    # Get distance to the wall
    distance_to_wall = get_distance_to_wall(true_state[:2], true_state[2], walls, screen)
    if distance_to_wall < 50:
        turn_time = np.random.random() * 10 + 10

    linear_velocity = 1
    angular_velocity = 0
    if turn_time > 0:
        turn_time -= dt
        linear_velocity = 0
        angular_velocity = 0.1

    

    control_inputs = np.array([[linear_velocity], [angular_velocity]])

    # Assuming control_inputs are [linear_acceleration, angular_acceleration]
    linear_acceleration, angular_acceleration = control_inputs[0, 0], control_inputs[1, 0]

    # Update velocities based on acceleration
    true_state[3] += linear_acceleration * dt  # Update linear velocity
    true_state[5] += angular_acceleration * dt # Update angular velocity

    # Convert orientation to radians for calculations
    orientation_rad = math.radians(true_state[2, 0])

    # Update position based on updated velocity
    true_state[0] += true_state[3] * dt * math.cos(orientation_rad) # Update x position
    true_state[1] += true_state[3] * dt * math.sin(orientation_rad) # Update y position

    # Update orientation based on angular velocity
    true_state[2] += math.degrees(true_state[5] * dt)  # Update orientation

    # Note: Ensure that the orientation is correctly wrapped around if needed


    
    # Create robot_rect for collision detection
    robot_rect = pygame.Rect(int(true_state[0, 0]) , int(true_state[1, 0]), 1, 1)


    if check_collision(robot_rect, walls):
        true_state[0] -= linear_velocity * dt * math.cos(math.radians(true_state[2]))
        true_state[1] -= linear_velocity * dt * math.sin(math.radians(true_state[2]))
        true_state[3] = 0
        true_state[4] = 0

    estx = kf.x[0, 0]
    esty = kf.x[1, 0]

    # Predict step
    kf.predict(control_inputs)

    # Update with position measurement
    if position_measurement_available(true_state):
        position_measurement = get_position_measurement(true_state)
        kf.set_measurement_matrix(H_position, R_position)
        kf.update(position_measurement)

    # Update with angle measurement
    if angle_measurement_available(true_state):
        angle_measurement = get_angle_measurement(true_state)
        kf.set_measurement_matrix(H_angle, R_angle)
        kf.update(angle_measurement)

    # estimated_rect = pygame.Rect(int(kf.x[0, 0])-25, int(kf.x[1, 0])-25, 50, 50)

    # # Check for collision with walls
    # if check_collision(estimated_rect, walls):
    #     # Handle collision
    #     # For example, you might reset the estimated position to its previous value
    #     kf.x[0, 0] = estx
    #     kf.x[1, 0] = esty


    # Visualization
    screen.fill((0, 0, 0))

    # Draw walls
    for wall in walls:
        wall.draw(screen)

    draw_robot(screen, true_state[:2], true_state[2], (0, 0, 255))  # True state
    draw_robot(screen, kf.x[:2], kf.x[2], (255, 0, 0))  # Estimated state
    draw_ellipse(screen, (0, 255, 0), kf.x[:2], kf.P)  # Uncertainty ellipse


    
    # Update the sensor line calculation
    orientation_rad = math.radians(true_state[2, 0])  # Ensure you use the current orientation
    sensor_start_x, sensor_start_y = int(true_state[0, 0]), int(true_state[1, 0])
    sensor_end_x = sensor_start_x + distance_to_wall * math.cos(orientation_rad)
    sensor_end_y = sensor_start_y + distance_to_wall * math.sin(orientation_rad)

    # Draw sensor line
    pygame.draw.line(screen, (255, 0, 0), (sensor_start_x, sensor_start_y), (int(sensor_end_x), int(sensor_end_y)), 1)

    # Displaying the current and estimated state
    true_state_text = f"True State: x={true_state[0,0]:.2f}, y={true_state[1,0]:.2f}, angle={math.degrees(math.atan2(math.cos(math.radians(true_state[2,0])),   math.sin(math.radians(true_state[2,0])))):.2f}, vx={true_state[3,0]:.2f}, vy={true_state[4,0]:.2f}, omega={true_state[5,0]:.2f}"
    estimated_state_text = f"Estimated: x={kf.x[0,0]:.2f}, y={kf.x[1,0]:.2f}, angle={math.degrees(math.atan2(math.cos(math.radians(kf.x[2,0])),   math.sin(math.radians(kf.x[2,0])))):.2f}, vx={kf.x[3,0]:.2f}, vy={kf.x[4,0]:.2f}, omega={kf.x[5,0]:.2f}"
    distance_text = f"Distance to wall: {distance_to_wall:.2f}"
    render_text(screen, true_state_text, (10, 10))
    render_text(screen, estimated_state_text, (10, 40))
    render_text(screen, distance_text, (10, 60))

    pygame.display.flip()

    pygame.time.delay(20)
    clock.tick(100)
    time += dt