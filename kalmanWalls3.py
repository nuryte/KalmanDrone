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
        self.max_covariance_value = 10

    def set_measurement_matrix(self, H, R):
        self.H = H
        self.R = R


    def predict(self, u=0, walls=None):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        for i in range(len(P)):
            for j in range(len(P[i])):
                P[i, j] = min(P[i, j], self.max_covariance_value)

        # Adjust if predicted position collides with any wall
        if walls is not None:
            for wall in walls:
                if self.adjust_for_collision(wall):
                    break

        return self.x

    def adjust_for_collision(self, wall):
        # Create a small rectangle to represent the robot's predicted position
        predicted_rect = pygame.Rect(int(self.x[0, 0]), int(self.x[1, 0]), 1, 1)

        # Check for collision and adjust the state
        if predicted_rect.colliderect(wall.rect):
            # Collision detected, adjust the state
            # Adjust position
            if self.x[3, 0] > 0 and self.x[0, 0] > wall.rect.right:  # Moving right, hit right wall
                self.x[0, 0] = wall.rect.right
                self.x[3, 0] = 0#-self.x[3, 0] * 0.5  # Reverse and reduce x velocity
                self.x[6, 0] = 0  # Reset x acceleration
            elif self.x[3, 0] < 0 and self.x[0, 0] < wall.rect.left:  # Moving left, hit left wall
                self.x[0, 0] = wall.rect.left
                self.x[3, 0] = 0#-self.x[3, 0] * 0.5  # Reverse and reduce x velocity
                self.x[6, 0] = 0  # Reset x acceleration

            if self.x[4, 0] > 0 and self.x[1, 0] > wall.rect.bottom:  # Moving down, hit bottom wall
                self.x[1, 0] = wall.rect.bottom
                self.x[4, 0] = 0#-self.x[4, 0] * 0.5  # Reverse and reduce y velocity
                self.x[7, 0] = 0  # Reset y acceleration
            elif self.x[4, 0] < 0 and self.x[1, 0] < wall.rect.top:  # Moving up, hit top wall
                self.x[1, 0] = wall.rect.top
                self.x[4, 0] = 0#-self.x[4, 0] * 0.5  # Reverse and reduce y velocity
                self.x[7, 0] = 0  # Reset y acceleration
            return True
        return False

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

    def update(self, z, H_custom, R_custom):
        """
        Update step with customized measurement model.
        z: measurement vector.
        H_custom: Custom measurement matrix.
        R_custom: Custom measurement noise covariance.
        """
        y = z - np.dot(H_custom, self.x)
        S = R_custom + np.dot(H_custom, np.dot(self.P, H_custom.T))
        K = np.dot(np.dot(self.P, H_custom.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(H_custom, self.P))

class Robot:
    def __init__(self, x, y, orientation, size=20, max_speed=1, max_turn_rate=0.1):
        # Initialize robot state
        self.position = np.array([x, y])
        self.orientation = orientation
        self.size = size
        self.max_speed = max_speed
        self.max_turn_rate = max_turn_rate

    def move(self, linear_velocity, angular_velocity, dt):
        # Limit the velocities
        linear_velocity = max(min(linear_velocity, self.max_speed), -self.max_speed)
        angular_velocity = max(min(angular_velocity, self.max_turn_rate), -self.max_turn_rate)

        # Update orientation
        self.orientation += angular_velocity * dt
        self.orientation %= 2 * math.pi  # Wrap around

        # Update position
        dx = linear_velocity * dt * math.cos(self.orientation)
        dy = linear_velocity * dt * math.sin(self.orientation)
        self.position += np.array([dx, dy])

    def draw(self, surface, color=(0, 0, 255)):
        # Calculate triangle vertices for drawing the robot
        angle = self.orientation
        points = [
            (self.position[0] + self.size * math.cos(angle), self.position[1] + self.size * math.sin(angle)),
            (self.position[0] + self.size * math.cos(angle + 2.2), self.position[1] + self.size * math.sin(angle + 2.2)),
            (self.position[0] + self.size * math.cos(angle - 2.2), self.position[1] + self.size * math.sin(angle - 2.2))
        ]
        pygame.draw.polygon(surface, color, points)

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


def draw_ellipse2(surface, color, position, covariance):
    """ Draw an ellipse representing the uncertainty in position. """
    # Ensure only the position covariance is used (top-left 2x2 submatrix)
    pos_covariance = covariance[:2, :2]
    width, height = np.sqrt(np.diag(pos_covariance)) * 2  # Scale factor for visualization
    width, height = int(width), int(height)  # Convert to integers
    ellipse_rect = pygame.Rect(int(position[0] - width / 2), int(position[1] - height / 2), width, height)
    pygame.draw.ellipse(surface, color, ellipse_rect, 1)

def draw_ellipse(surface, color, position, covariance, scale=2):
    # Flatten position array and convert to integer tuple
    position = tuple(map(int, position.flatten()))


    # Extract the position covariance (top-left 2x2 submatrix)
    pos_covariance = covariance[:2, :2]

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(pos_covariance)

    # Order eigenvalues and eigenvectors
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Width and height of ellipse based on eigenvalues
    width, height = np.sqrt(eigenvalues) * scale * 2  # Scale factor for visualization
    width, height = int(width), int(height)

    # Angle of rotation in radians
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # Create a surface large enough to accommodate the rotated ellipse
    max_dimension = max(width, height)
    ellipse_surface = pygame.Surface((max_dimension * 2, max_dimension * 2), pygame.SRCALPHA)
    ellipse_surface.fill((0, 0, 0, 0))  # Transparent background

    # Draw the ellipse on the surface (centered)
    ellipse_rect = pygame.Rect(max_dimension - width // 2, max_dimension - height // 2, width, height)
    pygame.draw.ellipse(ellipse_surface, color, ellipse_rect, 1)

    # Rotate the ellipse surface and blit onto the main surface
    rotated_surface = pygame.transform.rotate(ellipse_surface, np.degrees(-angle))

    # Adjusted line with error handling
    try:
        rotated_rect = rotated_surface.get_rect(center=position)
    except TypeError as e:
        print("Error in rotated_rect assignment:", e)
        print("Position value causing error:", position)
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
    return R_rotated


# Pygame Initialization
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Initialize robot state with expanded state vector
true_state = np.array([[400.0], [300.0], [275.0], [0.0], [0.0], [0.0]])  # Include velocity components
# Initialization
robot = Robot(400, 300, 0)


# Kalman Filter Setup with expanded state vector
dt = 0.1  # Time step
b_linear = 0.1    # Linear drag coefficient
b_angular = .5  # Angular drag coefficient

A = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],                # x_position updated
    [0, 1, 0, 0, 0, 0, 0, 0, 0],                # y_position updated
    [0, 0, 1, 0, 0, 0, 0, 0, 0],                # orientation updated
    [0, 0, 0, 0, 0, 0, 0, 0, 0],          # x_velocity updated with drag
    [0, 0, 0, 0, 0, 0, 0, 0, 0],          # y_velocity updated with drag
    [0, 0, 0, 0, 0, 0, 0, 0, 0],         # angular_velocity updated with drag
    [0, 0, 0, 0, 0, 0, 0, 0, 0],                           # x_acceleration
    [0, 0, 0, 0, 0, 0, 0, 0, 0],                           # y_accel
    [0, 0, 0, 0, 0, 0, 0, 0, 0]])                   # angular_acceleration (assumed constant for this timestep)

B = np.array([[dt, 0],
              [0, dt],
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

sigma_radial = 50.0  # Example value, adjust as needed
sigma_perpendicular = 20.0  # Example value, adjust as needed


initial_covariance_value = 10000

Q = np.eye(9) * 5 #increases how fast the uncertainty expands
R = np.eye(3) * 10 #unused but should give how good the measurment is
P = np.eye(9) * initial_covariance_value  # initial_covariance_value is a tuning parameter

kf = KalmanFilter(dt, A, B, H, Q, R, P)

kf.x[0] = true_state[0]
kf.x[1] = true_state[1]

def position_measurement_available(true_state):
    if np.random.random() > .95:
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
turn_time = 0
total_turn_time = 0
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
    if distance_to_wall < 80 and turn_time <= 0:
        turn_time = np.random.random() * 2 + 3 + 1
        total_turn_time = turn_time

    # linear_velocity = 0
    # if time < 10:
    linear_velocity = 10
    angular_velocity = 0
    if turn_time > 0:
        
        linear_velocity = 0
        angular_velocity = .3
        # else:
        #     linear_velocity = 0
        #     angular_velocity = -0.1

        turn_time -= dt

    

    control_inputs = np.array([[linear_velocity], [angular_velocity]])

    # Assuming control_inputs are [linear_acceleration, angular_acceleration]
    linear_acceleration, angular_acceleration = control_inputs[0, 0], control_inputs[1, 0]
    temp_true = true_state.copy()
    # Update velocities based on acceleration
    true_state[3]  = linear_acceleration#+= linear_acceleration * dt  # Update linear velocity
    #true_state[5] *= .99
    true_state[5] = angular_acceleration #* dt # Update angular velocity
    

    # Convert orientation to radians for calculations
    orientation_rad = math.radians(true_state[2, 0])

    # Update position based on updated velocity
    true_state[0] += true_state[3] * dt * math.cos(orientation_rad)  + .1 # Update x position
    true_state[1] += true_state[3] * dt * math.sin(orientation_rad) # Update y position

    # Update orientation based on angular velocity
    true_state[2] += math.degrees(true_state[5] * dt)  # Update orientation

    # Note: Ensure that the orientation is correctly wrapped around if needed


    
    # Create robot_rect for collision detection
    robot_rect = pygame.Rect(int(true_state[0, 0]) , int(true_state[1, 0]), 1, 1)


    if check_collision(robot_rect, walls):
        true_state[0] = temp_true[0]
        true_state[1] = temp_true[1]
        true_state[3] = 0

    estx = kf.x[0, 0]
    esty = kf.x[1, 0]
    orientation_rad2 = math.radians(kf.x[2, 0])

    # Predict step
    predict_control  = np.array([[linear_velocity * math.cos(orientation_rad2) ], [linear_velocity* math.sin(orientation_rad2)]])
    kf.predict(predict_control, walls=walls)

    # Update with position measurement
    # if position_measurement_available(true_state):
    #     position_measurement = get_position_measurement(true_state)
    #     R_postemp = rotate_covariance_matrix(orientation_rad, sigma_radial, sigma_perpendicular)
    #     # kf.set_measurement_matrix(H_position, R_postemp)
    #     kf.update(position_measurement, H_position, R_postemp)

    #distance_to_wall
    tof_measurement_noise_variance = 5000
    angle_strict = np.pi/8
    # Measurement matrix temp
    measure = True
    oriented_dist = 1
    orientation_rad = math.atan2(math.cos(math.radians(kf.x[2,0])),   math.sin(math.radians(kf.x[2,0]))) #normalize angle between -pi and pi
    if distance_to_wall < 100 and turn_time <= 0:#position_measurement_available(true_state):
        # Determine the measurement matrix H based on orientation
        if -np.pi/4 + angle_strict < orientation_rad < np.pi/4- angle_strict:  # Facing up
            # print("1", orientation_rad)
            H_tof = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0]])  # Measure y_position
            oriented_dist = 480 - np.cos(orientation_rad) * distance_to_wall#true_state[1] + np.random.normal(0, 5)#oriented_dist = np.cos(orientation_rad) * distance_to_wall
        elif -3*np.pi/4 + angle_strict < orientation_rad < -np.pi/4 - angle_strict:  # Facing left
            # print("2", orientation_rad)
            H_tof = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])  # Measure x_position
            oriented_dist = 120 - np.sin(orientation_rad) * distance_to_wall#true_state[0] + np.random.normal(0, 5)#np.sin(orientation_rad) * distance_to_wall
        elif np.pi/4 + angle_strict < orientation_rad < 3*np.pi/4- angle_strict:  # Facing right
            # print("3", orientation_rad)
            H_tof = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]])  # Measure x_position
            oriented_dist = 680 - np.sin(orientation_rad) * distance_to_wall#true_state[0] + np.random.normal(0, 5)#np.sin(orientation_rad) * distance_to_wall
        elif -3*np.pi/4 - angle_strict > orientation_rad  or orientation_rad > 3*np.pi/4 + angle_strict:  # Facing down
            # print("4", orientation_rad)
            H_tof = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0]])  # Measure y_position
            oriented_dist = 120 - np.cos(orientation_rad) * distance_to_wall#true_state[1] + np.random.normal(0, 5)#oriented_dist = np.cos(orientation_rad) * distance_to_wall
        else:
            # print("5", orientation_rad)
            measure = False
        
        if measure:
            # Measurement noise covariance
            R_tof = np.array([[tof_measurement_noise_variance]])

            # Update Kalman Filter with the distance measurement
            kf.update(oriented_dist, H_tof, R_tof)


    # Update with angle measurement
    if angle_measurement_available(true_state):
        angle_measurement = get_angle_measurement(true_state)
        # kf.set_measurement_matrix(H_angle, R_angle)
        kf.update(angle_measurement, H_angle, R_angle)

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