import numpy as np
import matplotlib.pyplot as plt

def generate_equidistant_points(points, num_points):
    num_original_points = len(points)
    interval_length = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)) / (num_points - num_original_points + 1)
    equidistant_points = [points[0]]  # Start with the first point

    for i in range(num_original_points - 1):
        p1 = points[i]
        p2 = points[i + 1]
        distance = np.linalg.norm(p2 - p1)
        num_equidistant_points = int(distance / interval_length)
        increment_value = (p2 - p1) / (num_equidistant_points + 1)

        for j in range(num_equidistant_points):
            new_point = p1 + increment_value * (j + 1)
            equidistant_points.append(new_point)

    equidistant_points.append(points[-1])  # Add the last point
    return np.array(equidistant_points)

# Example usage
given_points = np.array([[0, 0], [1, 1], [2, 3], [5, 2]])
equidistant_points = generate_equidistant_points(given_points, 10)
print(equidistant_points)

for (x,y) in given_points:
    plt.plot(x,y,'g*')

for (x,y) in equidistant_points:
    plt.plot(x,y,'r',marker='.')

plt.show()