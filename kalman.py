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
dt = 0.5  # Time step
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

initial_covariance_value =100

Q = np.eye(9) * .01
R = np.eye(3) * 1
P = np.eye(9) * initial_covariance_value  # initial_covariance_value is a tuning parameter

kf = KalmanFilter(dt, A, B, H, Q, R, P)

def position_measurement_available(true_state):
    if np.random.random() > .7:
        return True
    return False

def get_position_measurement(true_state):
    # Simulate Measurement with Some Noise for x and y position
    position_noise = np.random.normal(0, 5, size=(2, 1))  # Noise with standard deviation of 10
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
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Simulate Control Inputs (adapt as needed)
    if time %100 < 25:
        linear_velocity = 6
        angular_velocity = 0
    elif time %100 < 50:
        linear_velocity = 0
        angular_velocity = 5
    elif time %100 < 75:
        linear_velocity = 6
        angular_velocity = 0
    elif time %100 < 100:
        linear_velocity = 0
        angular_velocity = 5
    control_inputs = np.array([[linear_velocity], [angular_velocity]])

    # Update true state
    true_state[2] += angular_velocity * dt
    true_state[0] += linear_velocity * dt * math.cos(math.radians(true_state[2]))
    true_state[1] += linear_velocity * dt * math.sin(math.radians(true_state[2]))
    true_state[3] = linear_velocity * math.cos(math.radians(true_state[2]))
    true_state[4] = linear_velocity * math.sin(math.radians(true_state[2]))
    true_state[5] = angular_velocity

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

    # Visualization
    screen.fill((0, 0, 0))
    draw_robot(screen, true_state[:2], true_state[2], (0, 0, 255))  # True state
    draw_robot(screen, kf.x[:2], kf.x[2], (255, 0, 0))  # Estimated state
    draw_ellipse(screen, (0, 255, 0), kf.x[:2], kf.P)  # Uncertainty ellipse

    # Displaying the current and estimated state
    true_state_text = f"True State: x={true_state[0,0]:.2f}, y={true_state[1,0]:.2f}, angle={math.degrees(math.atan2(math.cos(math.radians(true_state[2,0])),   math.sin(math.radians(true_state[2,0])))):.2f}, vx={true_state[3,0]:.2f}, vy={true_state[4,0]:.2f}, omega={true_state[5,0]:.2f}"
    estimated_state_text = f"Estimated: x={kf.x[0,0]:.2f}, y={kf.x[1,0]:.2f}, angle={math.degrees(math.atan2(math.cos(math.radians(kf.x[2,0])),   math.sin(math.radians(kf.x[2,0])))):.2f}, vx={kf.x[3,0]:.2f}, vy={kf.x[4,0]:.2f}, omega={kf.x[5,0]:.2f}"
    render_text(screen, true_state_text, (10, 10))
    render_text(screen, estimated_state_text, (10, 40))

    pygame.display.flip()

    pygame.time.delay(100)
    clock.tick(30)
    time += dt