
from robot import *  # Check the robot.py tab to see how this works.
from math import *
from matrix import * # Check the matrix.py tab to see how this works.
import random
import copy

counter = 0

# ESTIMATION USING PARTICLE FILTER
def estimate_next_pos_particle(measurement, OTHER = None):
    """Estimate the next (x, y) position of the wandering Traxbot
    based on noisy (x, y) measurements."""
    global counter
    N = 10000
    
    
    if not OTHER: # this is the first measurement
        #print (measurement[0], measurement[1])
        x_meas = measurement[0]
        y_meas = measurement[1]
        
        abs_heading = 0#atan2(y_meas, x_meas)
        
        OTHER = (measurement,[abs_heading, 0, 0, 0])
        xy_estimate = OTHER[0]
        
    elif counter < 3:   # This is the algebraic solution ran for the first 3 trials

        dx = measurement[0] - OTHER[0][0]
        dy = measurement[1] - OTHER[0][1]
        
        x_meas = measurement[0]
        y_meas = measurement[1]
        
        #abs_heading = atan2(y_meas, x_meas)
        
        heading = atan2(dy,dx)
    
        d = dx / cos(heading)
        
        d_turning = heading - OTHER[1][0]
        # new heading
        abs_heading = heading + d_turning
    
        x = measurement[0] + d * cos(abs_heading)
        y = measurement[1] + d * sin(abs_heading)
    
        xy_estimate = (x,y)
    
        result_list = [heading, abs_heading, d_turning, d]
        
        OTHER = (measurement,result_list)
        
    elif counter == 3:      # This is initialization of the particle filter with best algebraic solution
        
        p_set = []
        weights = []
       
        std_dev = 0.2
        
        for i in range(N):
            heading_init = random.gauss(OTHER[1][1], OTHER[1][1]*std_dev)
            turning_init = random.gauss(OTHER[1][2], OTHER[1][2]*std_dev)
            distance_init = random.gauss(OTHER[1][3], OTHER[1][3]*std_dev)

            x_init = random.gauss(measurement[0], measurement[0]*std_dev)
            y_init = random.gauss(measurement[1], measurement[1]*std_dev)

            
            
            #print("heading: ", headix.xng_init, " turning: ", turning_init, " distance: ", distance_init)
            x = robot(x_init, y_init, heading_init, turning_init, distance_init, i)

            p_set.append(x)
            weights.append(0)
    
        new_p = []
        for idx in range(len(p_set)):
            new_p.append(p_set[idx].move_in_circle())
    
        # Assign set of particles to the OTHER variable to retain them in next cycle
        OTHER = []
        OTHER = new_p[:]
   
        chosen = OTHER[weights.index(max(weights))]
        
        # return the first particle
        xy_estimate = (chosen.sense())
    
    else:       # This is a runtime particle filter solution
        p_set, weights = particle_filter(OTHER,measurement)
        
        new_p = []
        
        for i in range(len(p_set)):
            new_p.append(p_set[i].move_in_circle())
    
        # Assign set of particles to the OTHER variable to retain them in next cycle
        OTHER = []
        OTHER = new_p[:]
   
        chosen = OTHER[weights.index(max(weights))]
        print("ID of the chosen: ", chosen.idx)
        # return the first particle
        xy_estimate = (chosen.sense())
    
    counter += 1
    return xy_estimate, OTHER 

# ESTIMATION USING KALMAN FILTER
def estimate_next_pos(measurement, OTHER = None):
    """Estimate the next (x, y) position of the wandering Traxbot
    based on noisy (x, y) measurements."""
    global counter
    
    if not OTHER: # this is the first measurement
        
        abs_heading = 0#atan2(y_meas, x_meas)
        
        OTHER = (measurement,[abs_heading, 0, 0, 0])
        xy_estimate = OTHER[0]
        
    elif counter < 3:   # This is the algebraic solution ran for the first 3 trials

        dx = measurement[0] - OTHER[0][0]
        dy = measurement[1] - OTHER[0][1]

        
        heading = atan2(dy,dx)
    
        d = dx / cos(heading)
        
        d_turning = heading - OTHER[1][0]
        # new heading
        abs_heading = heading + d_turning
    
        x = measurement[0] + d * cos(abs_heading)
        y = measurement[1] + d * sin(abs_heading)
    
        xy_estimate = (x,y)
    
        result_list = [heading, abs_heading, d_turning, d]
        
        OTHER = (measurement,result_list)
        
    elif counter == 3:      # This is initialization of the kalman filter
        measurement_history = [measurement]
        # initial state matrix
        x_init = matrix([[measurement[0]],#x
                         [measurement[1]],#y
                         [OTHER[1][1]], # heading
                         [OTHER[1][2]], # turning
                         [OTHER[1][3]]])# distance
        #state covariance matrix
        P = matrix([[1.0, 0.0, 0.0, 0.0, 0.0], 
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0]])
    
        Z = matrix([[measurement[0]],
                    [measurement[1]]])
    
        x, P = kalman_filter(x_init,P,Z)
        
        
        OTHER = []
        OTHER = (x, P, measurement_history)
        xy_estimate = (x.value[0][0], x.value[1][0])
    
    else:       # This is a runtime kalman filter solution 
        OTHER[2].append(measurement)
        history = []
        history = OTHER[2].copy()
        
        print("counter: ", counter)
        Z = matrix([[history[-1][0]],
                    [history[-1][1]]])
    
        x, P = kalman_filter(OTHER[0], OTHER[1], Z)
        
        
        OTHER = (x,P,history)
        xy_estimate = (x.value[0][0], x.value[1][0])
        
#        print("X: ", x.va)
        
    counter += 1
    return xy_estimate, OTHER



# Initial covariance matrix
Q = matrix([[100.0, 0.0, 0.0, 0.0, 0.0], 
            [0.0, 100.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0]])

# measurement matrix
H = matrix([[1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0]])
# measurement uncertainty
R = matrix([[0.0, 0.0],
            [0.0, 0.0]])
# identity matrix
I = matrix([[1.0, 0.0, 0.0, 0.0, 0.0], 
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]])


def kalman_filter(x,P,Z):
    
    d = x.value[4][0]
    ang = x.value[2][0] + x.value[3][0]
    
    #Jacobian
    F = matrix([[1.0, 0.0, -d*sin(ang), -d*sin(ang), cos(ang)], 
                [0.0, 1.0, d*cos(ang), d*cos(ang), sin(ang)],
                [0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0]])
    

    
    
    # measurement update
    y = Z - (H * x)
    S = H * P * H.transpose() + R
    K = P * H.transpose() * S.inverse()
    x = x + (K * y)
    
    SUP = (I - (K * H))
    P = (I - (K * H)) * P * SUP.transpose() + K * R * K.transpose()


    # angle truncation
    x.value[2][0] = angle_trunc(x.value[2][0])
    x.value[3][0] = angle_trunc(x.value[3][0])
    
    
#    P = matrix([[1/2]]) * (P + P.transpose())
    
    # prediction
    x = (F * x)
    P = F * P * F.transpose() + Q
    
    # angle truncation
    x.value[2][0] = angle_trunc(x.value[2][0])
    x.value[3][0] = angle_trunc(x.value[3][0])
    
    return (x,P)


def particle_filter(p, measurement):

    w = []

    for i in range(len(p)):
        particle_pos = (p[i].x, p[i].y)
        # The particles weight is a distance between current position and measurement
        a = 1/distance_between(particle_pos, measurement)
        w.append(a)

    p3 = []
#    p2 = []
    weights = []
    index = int(random.random() * len(p))
    beta = 0.0
    mw = max(w)
    for i in range(len(p)):
        beta += random.random() * 2.0 * mw
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % len(p)
        p3.append(copy.copy(p[index]))
        weights.append(w[index])
#        p2.append(copy.copy(p[i]))
    # return the particles that were selected in a weight wheel resampling
    return p3,weights
 

# A helper function 
def distance_between(point1, point2):
    """Computes distance between point1 and point2. Points are (x, y) pairs."""
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def simulate(estimate_next_pos_fcn, target_bot, OTHER = None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0
    # if you haven't localized the target bot, make a guess about the next
    # position, then we move the bot and compare your guess to the true
    # next position. When you are close enough, we stop checking.
    while not localized and ctr <= 1000:
        ctr += 1
        measurement = target_bot.sense()
        position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)
        error = distance_between(position_guess, true_position)
        if error <= distance_tolerance:
            print("You got it right! It took you ", ctr, " steps to localize.")
            localized = True
        if ctr == 1000:
            print("Sorry, it took you too many steps to localize the target.")
        print("Error: ", error, "Tolerance: ", distance_tolerance)  
    return localized

def simulate_vis(estimate_next_pos_fcn, target_bot, OTHER = None):
    localized = False
    distance_tolerance = 0.01 * target_bot.distance
    ctr = 0
    # if you haven't localized the target bot, make a guess about the next
    # position, then we move the bot and compare your guess to the true
    # next position. When you are close enough, we stop checking.
    #For Visualization
    import turtle    #You need to run this locally to use the turtle module
    window = turtle.Screen()
    window.bgcolor('white')
    size_multiplier= 25.0  #change Size of animation
    broken_robot = turtle.Turtle()
    broken_robot.shape('turtle')
    broken_robot.color('green')
    broken_robot.resizemode('user')
    broken_robot.shapesize(0.2, 0.2, 0.2)
    measured_broken_robot = turtle.Turtle()
    measured_broken_robot.shape('circle')
    measured_broken_robot.color('red')
    measured_broken_robot.resizemode('user')
    measured_broken_robot.shapesize(0.2, 0.2, 0.2)
    prediction = turtle.Turtle()
    prediction.shape('arrow')
    prediction.color('blue')
    prediction.resizemode('user')
    prediction.shapesize(0.2, 0.2, 0.2)
    prediction.penup()
    broken_robot.penup()
    measured_broken_robot.penup()
    #End of Visualization
    while not localized and ctr <= 1000:
        ctr += 1
        measurement = target_bot.sense()
        position_guess, OTHER = estimate_next_pos_fcn(measurement, OTHER)
        target_bot.move_in_circle()
        true_position = (target_bot.x, target_bot.y)
        error = distance_between(position_guess, true_position)
        if error <= distance_tolerance:
            print("You got it right! It took you ", ctr, " steps to localize.")
            localized = False
        if ctr == 1000:
            print ("Sorry, it took you too many steps to localize the target.")
        print("Error: ", error, "Tolerance: ", distance_tolerance)
        #More Visualization
        measured_broken_robot.setheading(target_bot.heading*180/pi)
        measured_broken_robot.goto(measurement[0]*size_multiplier, measurement[1]*size_multiplier-200)
        measured_broken_robot.stamp()
        broken_robot.setheading(target_bot.heading*180/pi)
        broken_robot.goto(target_bot.x*size_multiplier, target_bot.y*size_multiplier-200)
        broken_robot.stamp()
        prediction.setheading(target_bot.heading*180/pi)
        prediction.goto(position_guess[0]*size_multiplier, position_guess[1]*size_multiplier-200)
        prediction.stamp()
        #End of Visualization
#        if input("Press enter to continue") == 'q':
#            break
    window.bye()
    return localized


test_target = robot(2.1, 4.3, 0.5, 2*pi / 34.0, 1.5)
measurement_noise = 0.0 * test_target.distance
test_target.set_noise(0.0, 0.0, measurement_noise)

simulate_vis(estimate_next_pos, test_target)




