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
        # self.max_covariance_value = 10


    def predict(self, u=0, walls=None, Q = None, A = None):
        if Q is not None:
            self.Q = Q
        if A is not None:
            self.A = A
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return self.x


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
    wall_collide = None
    for wall in walls:
        # Check if line intersects with wall rectangle
        if wall.rect.clipline(start_point, end_point):
            # Pygame's clipline returns a list of points
            points = wall.rect.clipline(start_point, end_point)
            if points:
                distance = math.hypot(points[0][0] - start_x, points[0][1] - start_y)
                min_distance = min(min_distance, distance)
                wall_collide = wall 

    return min_distance, wall_collide


# Define walls in your game world
# walls = [Wall(100, 100, 600, 20), Wall(100, 480, 600, 20),
#          Wall(100, 120, 20, 360), Wall(680, 120, 20, 360)]
minx_wall = 100
maxx_wall = 600
miny_wall = 100
maxy_wall = 400
walls = [Wall(minx_wall, miny_wall, maxx_wall, 1), Wall(minx_wall, maxy_wall + miny_wall, maxx_wall, 1),
         Wall(minx_wall, miny_wall, 1, maxy_wall), Wall(maxx_wall + minx_wall, miny_wall, 1, maxy_wall)]

angle_noise = 0
wind_angle = -4*np.pi/3
wind_velocity = 1
drift_angle = 0#.3
drift_velocity = 0

drift_x = math.cos(drift_angle) * drift_velocity
drift_y = math.sin(drift_angle) * drift_velocity
wind_x = math.cos(wind_angle) * wind_velocity
wind_y = math.sin(wind_angle) * wind_velocity


def check_collision(robot_rect, walls):
    for wall in walls:
        if robot_rect.colliderect(wall.rect):
            return True
    return False

def move_robot(true_state, control_inputs, walls):

    # Assuming control_inputs are [linear_acceleration, angular_acceleration]
    linear_acceleration, angular_acceleration = control_inputs[0, 0], control_inputs[1, 0]
    temp_true = true_state.copy()
    # Update velocities based on acceleration
    true_state[3]  = linear_acceleration * (np.random.normal(1, .2)) #+= linear_acceleration * dt  # Update linear velocity
    #true_state[5] *= .99
    true_state[5] = angular_acceleration * (np.random.normal(1, .2))#* dt # Update angular velocity
    

    # Convert orientation to radians for calculations
    orientation_rad = math.radians(true_state[2, 0])

    # Update position based on updated velocity
    drift_x = math.cos(drift_angle + orientation_rad) * drift_velocity
    drift_y = math.sin(drift_angle + orientation_rad) * drift_velocity
    
    
    true_state[0] += true_state[3] * dt * math.cos(orientation_rad + angle_noise) + (drift_x + wind_x)* dt # Update x position
    true_state[1] += true_state[3] * dt * math.sin(orientation_rad + angle_noise) + (drift_y + wind_y )* dt # Update y position

    # Update orientation based on angular velocity
    true_state[2] += math.degrees(true_state[5] * dt) + np.random.normal(0, 2) * dt# Update orientation

    # Create robot_rect for collision detection
    robot_rect = pygame.Rect(int(true_state[0, 0]) , int(true_state[1, 0]), 1, 1)


    if check_collision(robot_rect, walls):
        true_state[0] = temp_true[0]
        true_state[1] = temp_true[1]
        true_state[3] = 0

    return true_state

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


def draw_ellipse(surface, color, position, covariance, scale=2):
    # Flatten position array and convert to integer tuple
    position = tuple(map(int, position.flatten()))


    # Extract the position covariance (top-left 2x2 submatrix)
    pos_covariance = covariance[:2, :2]

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(pos_covariance)
    # eigenvectors = np.real(eigenvectors)

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
    np.linalg.svd(R_rotated)
    return R_rotated


def kalRobotCoord(dt, Q, P, A):
    b_linear = 0.1    # Linear drag coefficient
    b_angular = .5  # Angular drag coefficient
    

                 

    B = np.array([[dt, 0, 0],
                [0, dt, 0],
                [0, 0, dt],
                [0, 0, 0],  # Control input 1 affects x_velocity
                [0, 0, 0],  # Control input 2 affects y_velocity
                [0, 0, 0],  
                [0, 0, 0],  
                [0, 0, 0],  
                ])  # Assuming no direct control input for angular velocity


    # Measurement matrix temp
    H = np.array([[1, 0, 0, 0, 0, 0,0,0],  # Measure x_position
                        [0, 1, 0, 0, 0, 0,0,0]]) # Measure y_position


    #Q = np.eye(8) * .001 #increases how fast the uncertainty expands
    R = np.eye(3) * 10 #unused but should give how good the measurment is
    #P = np.eye(8) * initial_covariance_value  # initial_covariance_value is a tuning parameter

    kf = KalmanFilter(dt, A, B, H, Q, R, P)
    return kf

def position_measurement_available(true_state):
    if np.random.random() > .95:
        return True
    return False

def get_position_measurement(true_state):
    # Simulate Measurement with Some Noise for x and y position
    position_noise = np.random.normal(0, 20, size=(2, 1))  # Noise with standard deviation of 10
    measured_position = true_state[:2] + position_noise

    return measured_position

def angle_measurement_available(true_state):
    if np.random.random() > .1:
        return True
    return False

def get_angle_measurement(true_state):
    angle_measurement = np.array([[true_state[2, 0]]]) + np.random.normal(0, 2)
    return angle_measurement

#setup


# Pygame Initialization
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()

# Initialize robot state with expanded state vector
true_state = np.array([[400.0], [300.0], [90.0], [0.0], [0.0], [0.0]])  # Include velocity components

# Measurement matrix for position (assuming measurements are x and y position)
H_position = np.array([[1, 0, 0, 0, 0, 0,0,0],  # Measure x_position
                    [0, 1, 0, 0, 0, 0,0,0]]) # Measure y_position

# Measurement matrix for angle (assuming measurement is orientation)
H_angle = np.array([[0, 0, 1, 0, 0,0,0,0]])    # Measure orientation

angle_measurement_noise_variance = .001

R_angle = np.array([[angle_measurement_noise_variance]])

sigma_radial = 100.0  # Example value, adjust as needed
sigma_perpendicular = 20.0  # Example value, adjust as needed

# Kalman Filter Setup with expanded state vector
dt = 0.2  # Time step

initial_covariance_value = 20
Q = np.eye(8) * .001 #increases how fast the uncertainty expands
Q[3][3] = 0.0001
Q[4][4] = 0.0001
Q[6][6] = 0.00001
Q[7][7] = 0.00001
P = np.eye(8) * 20  # initial_covariance_value is a tuning parameter
A = np.array([
    [1, 0, 0, dt, 0, 0,0,0],                # x_position updated (need to apply rotation on the drift (6,7))
    [0, 1, 0, 0, dt, 0,0,0],                # y_position updated (need to apply rotation on the drift (6,7))
    [0, 0, 1, 0, 0, dt,0,0],                # orientation updated
    [0, 0, 0, 1, 0, 0,0,0],          # x_velocity updated with drag
    [0, 0, 0, 0, 1, 0,0,0],          # y_velocity updated with drag
    [0, 0, 0, 0, 0, 0,0,0],          # angular velocity updated with drag
    [0, 0, 0, 0, 0, 0,1,0] ,         # x control abnormalities
    [0, 0, 0, 0, 0, 0,0,1]           # y control abnormalities
    ])  

net_rotate_estimation = np.zeros((2,2))
# Main Loop

kf = kalRobotCoord(dt, Q, P, A)

tof_measurement_noise_variance = 10000
# control_measurement_noise_variance = 10000
# Measurement noise covariance
R_tof = np.array([[tof_measurement_noise_variance]])
# R_control = np.array([[control_measurement_noise_variance]])
angle_strict = np.pi/8


kf.x[0] = true_state[0]
kf.x[1] = true_state[1]

detection_distance = 125
time = 0
turn_time = 0
total_turn_time = 0
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


    distance_to_wall, _ = get_distance_to_wall(true_state[:2], true_state[2], walls, screen)
    _, wall_collide = get_distance_to_wall(kf.x[:2], kf.x[2], walls, screen)
    if distance_to_wall < 80 and turn_time <= 0:
        turn_time = np.random.random() * 2 + 2 #+ 2
        total_turn_time = turn_time

    linear_velocity = 20
    angular_velocity = 0
    if turn_time > 0:
        linear_velocity = 0
        angular_velocity = .3
        turn_time -= dt

    

    control_inputs = np.array([[linear_velocity], [angular_velocity]])


    true_state = move_robot(true_state, control_inputs, walls)


    
    estx = kf.x[0, 0]
    esty = kf.x[1, 0]
    
    orientation_rad2 = math.radians(kf.x[2, 0])
    orientation_rad = math.atan2(math.sin(orientation_rad2),   math.cos(orientation_rad2)) #normalize angle between -pi and pi
    # print(orientation_rad)

    # Predict step
    predict_control  = np.array([[linear_velocity * math.cos(orientation_rad2) ], [linear_velocity* math.sin(orientation_rad2)], [angular_velocity]])
    
    rotate_estimation = np.array([[np.cos(orientation_rad), -np.sin(orientation_rad)], 
                    [np.sin(orientation_rad), np.cos(orientation_rad)]])
    # net_rotate_estimation += rotate_estimation * dt
    
    Q_rotate = rotate_covariance_matrix(orientation_rad, 1, .01) 
    # # print(Q_rotate, "1")
    Q[0:2, 0:2] = Q_rotate 
    A[0:2, 6:8] = rotate_estimation
    # print(A)
    kf.predict(predict_control, walls=walls, Q = Q, A = A)

    

    # Update with position measurement
    # if position_measurement_available(true_state):
    position_measurement = get_position_measurement(true_state)
    R_postemp = np.eye(2) * 1000#rotate_covariance_matrix(orientation_rad, sigma_radial, sigma_perpendicular)
    kf.update(position_measurement, H_position, R_postemp)

    #distance_to_wall
    # Measurement matrix temp
    # measure = True
    # oriented_dist = 1
    # pos_error = 0
    # if distance_to_wall < detection_distance:# and turn_time <= 0:#position_measurement_available(true_state):
        
    #     if wall_collide is not None:
    #         if wall_collide.rect.width > wall_collide.rect.height:
    #             H_tof = np.array([[0, 1, 0, 0, 0, 0,0,0]])  # Measure y_position
                
    #             oriented_dist = wall_collide.rect.y - np.sin(orientation_rad) *  np.random.normal(distance_to_wall, distance_to_wall * .2)

    #             # H_control = np.array([[0, 0, 0, 0, 0, 0,net_rotate_estimation[0,0],net_rotate_estimation[0,1]]]) 
    #             # net_rotate_estimation[0,0] = 0
    #             # net_rotate_estimation[0,1] = 0
    #             # pos_error  = (oriented_dist - kf.x[1,0]) # error in absolute y position
    #             #print(round(time,1), round(pos_error,1), round(kf.x[6,0],2), round(kf.x[7,0],2),round(H_control[0,6],2),round(H_control[0,7],2) ,round(net_rotate_estimation[1,0],2),round(net_rotate_estimation[1,1],2))
    #             #need to use this error along with orientation angle
                
    #         else:
    #             H_tof = np.array([[1, 0, 0, 0, 0, 0,0,0]])  # Measure x_position
    #             # if wall_collide.rect.y < wall_collide.rect.height:
    #             oriented_dist = wall_collide.rect.x - np.cos(orientation_rad) *  np.random.normal(distance_to_wall, distance_to_wall * .2)
                
    #             # H_control = np.array([[0, 0, 0, 0, 0, 0,net_rotate_estimation[1,0],net_rotate_estimation[1,1]]]) 
    #             # net_rotate_estimation[1,0] = 0
    #             # net_rotate_estimation[1,1] = 0
    #             # pos_error  = (oriented_dist - kf.x[0,0]) #error in absolute x position
    #             #print(round(time,1), round(pos_error,1), round(kf.x[6,0],2), round(kf.x[7,0],2),round(net_rotate_estimation[0,0],2),round(net_rotate_estimation[0,1],2),round(H_control[0,6],2),round(H_control[0,7],2) )
    #     else:
    #             measure = False

    #     if measure:

    #         # Update Kalman Filter with the distance measurement
    #         kf.update(oriented_dist, H_tof, R_tof)
            
    #         # Update Kalman Filter with the control measurement
    #         #kf.update(pos_error, H_control, R_control)


    # Update with angle measurement
    if angle_measurement_available(true_state):
        angle_measurement = get_angle_measurement(true_state)
        
        kf.update(angle_measurement, H_angle, R_angle)


    # Visualization
    screen.fill((0, 0, 0))

    # Draw walls
    for wall in walls:
        wall.draw(screen)

    draw_robot(screen, true_state[:2], true_state[2], (0, 0, 255))  # True state
    draw_robot(screen, kf.x[:2], kf.x[2], (255, 0, 0))  # Estimated state
    draw_ellipse(screen, (0, 255, 0), kf.x[:2], kf.P )  # Uncertainty ellipse


    
    # Update the sensor line calculation
    orientation_rad = math.radians(true_state[2, 0])  # Ensure you use the current orientation
    sensor_start_x, sensor_start_y = int(true_state[0, 0]), int(true_state[1, 0])
    sensor_end_x = sensor_start_x + min(distance_to_wall, detection_distance) * math.cos(orientation_rad)
    sensor_end_y = sensor_start_y + min(distance_to_wall, detection_distance) * math.sin(orientation_rad)

    # Draw sensor line
    pygame.draw.line(screen, (255, 0, 0), (sensor_start_x, sensor_start_y), (int(sensor_end_x), int(sensor_end_y)), 1)

    # Displaying the current and estimated state
    true_state_text = f"True State: x={true_state[0,0]:.2f}, y={true_state[1,0]:.2f}, angle={math.degrees(math.atan2(math.cos(math.radians(true_state[2,0])),   math.sin(math.radians(true_state[2,0])))):.2f}, vx={ wind_x:.2f}, vy={ wind_y:.2f}"
    estimated_state_text = f"Estimated: x={kf.x[0,0]:.2f}, y={kf.x[1,0]:.2f}, angle={math.degrees(math.atan2(math.cos(math.radians(kf.x[2,0])),   math.sin(math.radians(kf.x[2,0])))):.2f}, vx={kf.x[3,0]:.2f}, vy={kf.x[4,0]:.2f}"
    # control_text = f"control_error: x={(drift_x - kf.x[6,0]/dt):.2f}, y={(drift_y - kf.x[7,0]/dt):.2f} "
    control_text = f"Estimated drift: x={( kf.x[6,0]/ dt):.2f}, y={ (kf.x[7,0]/dt):.2f}, True drift: x={( drift_x):.2f}, y={ (drift_y):.2f}  "
    distance_text = f"Distance to wall: {distance_to_wall:.2f}"
    render_text(screen, true_state_text, (10, 10))
    render_text(screen, estimated_state_text, (10, 40))
    render_text(screen, control_text, (10, 60))

    pygame.display.flip()

    pygame.time.delay(20)
    clock.tick(100)
    time += dt