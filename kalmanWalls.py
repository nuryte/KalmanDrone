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




# Pygame Initialization
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Initialize robot state with expanded state vector
true_state = np.array([[400.0], [300.0], [0.0], [0.0], [0.0], [0.0]])  # Include velocity components


# Kalman Filter Setup with expanded state vector
dt = 0.5  # Time step
A = np.array([[1, 0, 0, dt,  0,  0],
              [0, 1, 0,  0, dt,  0],
              [0, 0, 1,  0,  0, dt],
              [0, 0, 0,  1,  0,  0],
              [0, 0, 0,  0,  1,  0],
              [0, 0, 0,  0,  0,  1]])
B = np.zeros((6, 2))  # Define based on your control inputs

# Measurement matrix temp
H = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0]])
# Measurement matrix for position
H_position = np.array([[1, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 0]])

# Measurement matrix for angle
H_angle = np.array([[0, 0, 1, 0, 0, 0]])
position_measurement_noise_variance = 2000
angle_measurement_noise_variance = 1
R_position = np.eye(2) * position_measurement_noise_variance
R_angle = np.array([[angle_measurement_noise_variance]])

Q = np.eye(6) * 1
R = np.eye(3) * 100
P = np.eye(6) * 100
kf = KalmanFilter(dt, A, B, H, Q, R, P)

def position_measurement_available(true_state):
    if np.random.random() > .9:
        return True
    return False

def get_position_measurement(true_state):
    # Simulate Measurement with Some Noise for x and y position
    position_noise = np.random.normal(0, 10, size=(2, 1))  # Noise with standard deviation of 10
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
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Simulate Control Inputs (adapt as needed)
    linear_velocity = 3
    angular_velocity = 5
    control_inputs = np.array([[linear_velocity], [angular_velocity]])

    # Update true state
    true_state[2] += angular_velocity * dt
    true_state[0] += linear_velocity * dt * math.cos(math.radians(true_state[2]))
    true_state[1] += linear_velocity * dt * math.sin(math.radians(true_state[2]))
    true_state[3] = linear_velocity
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
    pygame.display.flip()

    pygame.time.delay(100)
    clock.tick(30)