import math
import random
import numpy as np
import matplotlib.pyplot as plt

v0 = 50
h = 100
g = 9.81
margin_of_error = 5

def calculate_distance(angle_degrees):
    angle_radians = math.radians(angle_degrees)
    
    # Calculate time of flight using the formula for when the projectile hits the ground
    term_under_sqrt = v0**2 * (math.sin(angle_radians))**2 + 2 * g * h
    time_of_flight = (v0 * math.sin(angle_radians) + math.sqrt(term_under_sqrt)) / g
    
    distance = v0 * math.cos(angle_radians) * time_of_flight
    return distance, time_of_flight

def plot_trajectory(angle_degrees, time_of_flight):
    angle_radians = math.radians(angle_degrees)
    
    # Time values for plotting the trajectory
    time_values = np.linspace(0, time_of_flight, num=500)
    
    # Horizontal and vertical positions as a function of time
    x_values = v0 * np.cos(angle_radians) * time_values
    y_values = h + v0 * np.sin(angle_radians) * time_values - 0.5 * g * time_values**2
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, color='blue')
    
    plt.title('Projectile Motion for the Warwolf')
    plt.xlabel('Distance (m)')
    plt.ylabel('Height (m)')
    plt.grid(True)
    plt.savefig('trajectory.png')
    
    plt.show()

def main():
    target_distance = random.randint(50, 340)
    print(f"Target distance: {target_distance} meters")
    
    attempts = 0
    
    while True:
        try:
            angle = float(input("Enter the angle of the shot in degrees: "))
            
            distance, time_of_flight = calculate_distance(angle)
            print(f"Your projectile traveled {distance:.2f} meters.")
            
            if abs(distance - target_distance) <= margin_of_error:
                print(f"Target hit! You destroyed the target in {attempts + 1} attempts.")
                plot_trajectory(angle, time_of_flight)
                break
            else:
                print(f"Missed! Try again. The target is at {target_distance} meters.\n")
            
            attempts += 1
        
        except ValueError:
            print("Invalid input. Please enter a valid number for the angle.")

if __name__ == "__main__":
    main()
