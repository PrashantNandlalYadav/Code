-- Machine Learning

-- Practical_1 --

#pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt

# Define the range for the x-axis
x = np.linspace(-10, 10, 1000)

# Define all activation functions using lambda functions
activation_functions = {
    "Linear": lambda x: x,
    "Binary Step": lambda x: np.where(x >= 0, 1, 0),
    "Ramp": lambda x: np.maximum(0, x),
    "Gaussian": lambda x: np.exp(-x**2),
    "Sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "ReLU": lambda x: np.maximum(0, x),
    "Leaky ReLU": lambda x: np.where(x > 0, x, 0.01 * x),
    "Tanh": lambda x: np.tanh(x)
}

# Plot all activation functions
plt.figure(figsize=(12, 12))

for i, (name, func) in enumerate(activation_functions.items(), 1):
    plt.subplot(3, 3, i)
    plt.plot(x, func(x), label=name)
    plt.title(f"{name} Activation Function")
    plt.grid(True)

plt.tight_layout()
plt.show()


-- Practical_2 --

#pip install numpy tensorflow scikit-learn matplotlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate and preprocess dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, random_state=42)

# Build model with an explicit Input layer
model = Sequential([
    Input(shape=(20,)),  # Explicit input layer
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Plot accuracy and loss
for metric in ['accuracy', 'loss']:
    plt.plot(history.history[metric], label=f'train {metric}')
    plt.plot(history.history[f'val_{metric}'], label=f'test {metric}')
    plt.title(f'Model {metric.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()



-- Practical_3 --

#pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, lr=0.1, n_iters=10):
        # Initialize weights and bias
        self.w, self.b = np.zeros(2), 0
        self.lr, self.n_iters = lr, n_iters

    def fit(self, X, y):
        # Train the perceptron over multiple iterations
        for i in range(self.n_iters):
            # Calculate update values
            update = self.lr * (y - (X @ self.w + self.b >= 0).astype(int))
            # Update weights and bias
            self.w += update @ X
            self.b += update.sum()
            # Print detailed information per iteration
            print(f"Iteration {i + 1}/{self.n_iters}")
            print(f"Weights: {self.w}, Bias: {self.b}")

    def predict(self, X):
        # Predict using the step function
        return (X @ self.w + self.b >= 0).astype(int)

# Example usage
if __name__ == "__main__":
    # Define training data for OR logic gate
    X, y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])

    # Instantiate and train Perceptron
    p = Perceptron()
    print("\nTraining the Perceptron:")
    p.fit(X, y)

    # Make predictions
    predictions = p.predict(X)
    print("\nPredictions on training data:")
    for xi, target, pred in zip(X, y, predictions):
        print(f"Input: {xi}, Target: {target}, Prediction: {pred}")

    # Plot decision boundary
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', s=100)
    
    # Create decision boundary line
    x_values = np.linspace(-0.5, 1.5, 100)
    y_values = -(p.w[0] * x_values + p.b) / p.w[1]
    plt.plot(x_values, y_values, color='black', label='Decision Boundary')

    # Enhance plot details
    plt.title("Perceptron Decision Boundary for OR Logic Gate")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.grid()
    plt.legend()
    plt.show()

-- Practical_4 --


#pip install numpy matplotlib seaborn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
grid_size, goal_state = 5, (4, 4)
actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
Q = np.zeros((grid_size, grid_size, len(actions)))

# Training with visualization updates
for episode in range(1000):
    state = (0, 0)  # Start position
    path = [state]  # Track the agent's path for visualization

    while state != goal_state:
        # Choose action using epsilon-greedy strategy
        action = np.random.choice(len(actions)) if np.random.rand() < 0.1 else np.argmax(Q[state])
        
        # Move to the new state
        new_state = (max(0, min(state[0] + actions[action][0], grid_size - 1)),
                     max(0, min(state[1] + actions[action][1], grid_size - 1)))
        
        # Calculate reward
        reward = 1 if new_state == goal_state else -0.01  # Small penalty for each step

        # Q-learning update rule
        Q[state][action] += 0.1 * (reward + 0.9 * np.max(Q[new_state]) - Q[state][action])
        
        # Update state and path
        state = new_state
        path.append(state)

# Final Q-values visualization
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Visualize the final path taken by the agent
ax1 = axs[0]
for (x, y) in path:
    ax1.plot(y, x, 'bo')  # Mark the agent's path
ax1.plot(0, 0, 'go', markersize=10, label="Start")  # Start state
ax1.plot(goal_state[1], goal_state[0], 'ro', markersize=10, label="Goal")  # Goal state
ax1.set_title("Agent's Path During Training")
ax1.legend()
ax1.grid()
ax1.set_xlim(-0.5, grid_size - 0.5)
ax1.set_ylim(-0.5, grid_size - 0.5)
ax1.invert_yaxis()

# Visualize the Q-values as a heatmap for each action
ax2 = axs[1]
q_values_max = np.max(Q, axis=2)  # Max Q-values for each state
sns.heatmap(q_values_max, annot=True, cmap="YlGnBu", ax=ax2, cbar=True)
ax2.set_title("Max Q-values Heatmap")
ax2.invert_yaxis()

plt.show()

# Detailed Explanation
print("Detailed Q-values:")
print(Q)

-- Practical_5 --

#pip install matplotlib

# Aim :- Roulettes wheel to select fittest character string population.

import random
import matplotlib.pyplot as plt

# Define the genes and target
genes = ["I", "N", "D", "A"]
target = ["I", "N", "D", "I", "A"]

# Step 1: Generate a Random Initial Population
population = [[''.join(random.choices(genes, k=5))] for _ in range(20)]
print("Initial Population (20 individuals):")
for i, individual in enumerate(population, 1):
    print(f"{i}. {individual[0]}")

# Step 2: Define Fitness and Roulette Selection Function
def roulette(population):
    # Calculate fitness as the number of correct characters in each individual
    fitness = [sum(i == j for i, j in zip(individual[0], target)) for individual in population]
    
    # Display fitness values
    print("\nFitness Scores:")
    for ind, fit in zip(population, fitness):
        print(f"Individual: {ind[0]} - Fitness: {fit}")

    # Calculate selection probabilities for each individual
    prob = [f / sum(fitness) for f in fitness]

    # Visualization: Pie Chart with Exploded Slices
    plt.figure(figsize=(12, 12))  # Increase figure size
    explode = [0.1] * len(population)  # Explode all slices
    wedges, texts, autotexts = plt.pie(
        prob, 
        labels=[ind[0] for ind in population], 
        autopct="%1.1f%%", 
        startangle=140,
        explode=explode,  # Use explode to separate the slices
        textprops={'fontsize': 6},  # Adjust text size for labels (smaller)
        shadow=True,  # Add shadow for 3D effect
        labeldistance=1.3  # Increase label distance
    )
    plt.setp(autotexts, size=8, weight="bold", color="white")  # Smaller percentages
    plt.title("Selection Probabilities for Each Individual (Pie Chart)", fontsize=18, y=1.1)  # Adjusted y position higher
    plt.show()

    # Visualization: Bar Chart
    plt.figure(figsize=(10, 6))
    individuals = [ind[0] for ind in population]
    plt.barh(individuals, prob, color='skyblue')
    plt.xlabel("Selection Probability", fontsize=14)
    plt.ylabel("Individual", fontsize=14)
    plt.title("Selection Probabilities for Each Individual (Bar Chart)", fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # Add grid for better readability
    plt.show()

    # Select parents based on their probability weights
    selected_parents = random.choices(population, weights=prob, k=4)
    
    print("\nSelected Fittest Parents (for next generation):")
    for i, parent in enumerate(selected_parents, 1):
        print(f"Parent {i}: {parent[0]}")
    
    return selected_parents

# Step 3: Run the Roulette Selection
print("\nRunning Roulette-Wheel Selection...")
parents = roulette(population)








#Aim :- Roulettes wheel to select fittest binary string population.

#pip install numpy matplotlib


import numpy as np
import random
import matplotlib.pyplot as plt

def roulette(pop):
    # Calculate fitness scores as the sum of the individual's genes (0s and 1s)
    fitness = [sum(ind) for ind in pop]
    
    # Display fitness scores for each individual
    print("\nFitness Scores:")
    for index, ind in enumerate(pop):
        print(f"Individual {index}: {ind} - Fitness: {fitness[index]}")
    
    # Plot pie chart for fitness distribution
    plt.figure(figsize=(10, 10))  # Increase figure size for better visibility
    plt.pie(fitness, labels=[f"Ind {i}: {ind} (Fitness: {fit})" for i, (ind, fit) in enumerate(zip(pop, fitness))],
            autopct="%1.1f%%", startangle=140, shadow=True, textprops={'fontsize': 10})
    plt.title("Fitness Distribution of Population", fontsize=16)
    plt.show()
    
    # Select the fittest parents using roulette selection
    selected_parents = random.choices(pop, weights=fitness, k=2)
    
    return selected_parents

# Initialize a random population
pop = [[random.randint(0, 1) for _ in range(4)] for _ in range(12)]
print(f"Initial Random Population (12 individuals):")
for index, individual in enumerate(pop):
    print(f"Individual {index}: {individual}")

# Run the roulette selection
print("\nRunning Roulette-Wheel Selection...")
new_pop = roulette(pop)

# Display the selected fittest parents
print("The fittest parents selected are:")
for i, parent in enumerate(new_pop, 1):
    print(f"Parent {i}: {parent}")





-- Practical_6 --

#pip install numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 6]

def draw(pop, virus):
    plt.scatter(pop[:, 0], pop[:, 1], c='green', s=12)
    plt.scatter(*virus, c='red', s=60)
    plt.xlim(-100, 100); plt.ylim(-100, 100); plt.legend(['Population', 'Virus'])

pop = np.random.randint(-100, 100, (1000, 2)); virus = [5, 5]
print("Initial (first 5):", pop[:5], "\nVirus:", virus)

for _ in range(1):
    fittest = np.argsort(np.sum((pop - virus) ** 2, axis=1))[:100]
    print("\nFittest (indexes):", fittest, "\nDistances:", np.sum((pop[fittest] - virus) ** 2, axis=1))
    pop = np.array([np.concatenate((pop[np.random.choice(fittest)][:1], pop[np.random.choice(fittest)][1:])) for _ in range(1000)]) + np.random.randint(-10, 10, (1000, 2))

draw(pop, virus)
plt.show()
print("\nFinal (first 5):", pop[:5])


-- Practical_7 --

#pip install numpy matplotlib

import numpy as np

def aco_tsp(num_ants, num_iterations, decay, alpha, beta, distances):
    num_cities = len(distances)
    pheromones = np.ones((num_cities, num_cities))  # Pheromone levels initialized to 1

    def route_length(route):
        return sum(distances[route[i-1]][route[i]] for i in range(num_cities))

    def update_pheromones(pheromones, ants, best_route, best_length):
        pheromones *= (1 - decay)  # Evaporate pheromones
        for ant in ants:
            for i in range(num_cities):
                pheromones[ant[i-1]][ant[i]] += 1 / route_length(ant)  # Update pheromones based on ant's route
        for i in range(num_cities):
            pheromones[best_route[i-1]][best_route[i]] += 1 / best_length  # Reinforce the best route

    best_route = None
    best_length = np.inf

    for _ in range(num_iterations):
        ants = []
        for _ in range(num_ants):
            route = [np.random.randint(num_cities)]  # Start from a random city
            for _ in range(num_cities - 1):
                i = route[-1]
                # Use a small constant (e.g., 1e-6) for zero distances to avoid division by zero
                adjusted_distances = np.where(distances[i] == 0, 1e-6, distances[i])
                probabilities = pheromones[i] ** alpha * (1 / adjusted_distances) ** beta
                probabilities[route] = 0  # Prevent revisiting the same city
                next_city = np.random.choice(range(num_cities), p=probabilities / probabilities.sum())
                route.append(next_city)
            ants.append(route)

        for ant in ants:
            length = route_length(ant)
            if length < best_length:
                best_length = length
                best_route = ant
        
        update_pheromones(pheromones, ants, best_route, best_length)

    return best_route, best_length

# Example Usage
num_cities = 5
distances = np.random.rand(num_cities, num_cities) * 100  # Create a random distance matrix
np.fill_diagonal(distances, 0)  # Set distances from a city to itself to zero

best_route, best_length = aco_tsp(10, 100, 0.5, 1.0, 2.0, distances)

# Detailed Output
print("Distance Matrix:")
print(distances)
print("\nBest Route Found (city indices):", best_route)
print("Total Distance of Best Route:", best_length)

# Display distances for the best route
route_distances = [distances[best_route[i-1]][best_route[i]] for i in range(num_cities)]
print("\nDistances for Best Route:")
for i in range(num_cities):
    print(f"From City {best_route[i-1]} to City {best_route[i]}: {route_distances[i]:.2f}")

print("\nTotal Distance Calculation:")
print(" + ".join(f"{route_distances[i]:.2f}" for i in range(num_cities)))
print(f"= {best_length:.2f}")


-- Practical_8 --

#pip install numpy matplotlib

import numpy as np

# Convolution, ReLU, MaxPool, and Fully Connected in a concise form
class SimpleCNN:
    def __init__(self):
        # Initialize filters and weights with small random values
        self.conv_filters = np.random.randn(8, 3, 3) / 9  # 8 filters of size 3x3
        self.fc_weights = np.random.randn(13 * 13 * 8, 10) / (13 * 13 * 8)  # Fully connected weights
        self.fc_biases = np.zeros(10)  # Biases for the fully connected layer

    def conv(self, image):
        h, w = image.shape
        output = np.zeros((h - 2, w - 2, 8))  # Output shape after convolution
        for f in range(8):  # For each filter
            for i in range(h - 2):
                for j in range(w - 2):
                    output[i, j, f] = np.sum(image[i:i+3, j:j+3] * self.conv_filters[f])  # Convolution operation
        return output

    def relu(self, x):
        return np.maximum(0, x)  # ReLU activation function

    def maxpool(self, x):
        h, w, f = x.shape
        # Max pooling operation
        return np.array([[np.max(x[i:i+2, j:j+2], axis=(0, 1)) for j in range(0, w, 2)] for i in range(0, h, 2)])

    def fully_connected(self, x):
        return np.dot(x.flatten(), self.fc_weights) + self.fc_biases  # Fully connected layer operation

    def softmax(self, x):
        exp_vals = np.exp(x - np.max(x))  # Subtracting max for numerical stability
        return exp_vals / np.sum(exp_vals)  # Softmax function to convert to probabilities

    def forward(self, image):
        # Step 1: Input Image
        print(f"Step 1 - Input Image: Shape {image.shape}\n")
        
        # Step 2: Convolution Layer
        conv_output = self.conv(image)
        print(f"Step 2 - After Convolution: Applied 8 filters of size 3x3. Output Shape: {conv_output.shape}")
        print(f"Sample Output (First Filter, top-left 2x2 patch):\n{conv_output[:2, :2, 0]}\n")
        
        # Step 3: ReLU Activation
        relu_output = self.relu(conv_output)
        print(f"Step 3 - After ReLU: Replaced negative values with zero. Output Shape: {relu_output.shape}")
        print(f"Sample Output (First Filter, top-left 2x2 patch after ReLU):\n{relu_output[:2, :2, 0]}\n")
        
        # Step 4: Max Pooling
        pool_output = self.maxpool(relu_output)
        print(f"Step 4 - After MaxPooling: Reduced each 2x2 region to its maximum value. Output Shape: {pool_output.shape}")
        print(f"Sample Output (First Filter, top-left 2x2 region after pooling):\n{pool_output[:2, :2, 0]}\n")
        
        # Step 5: Fully Connected Layer
        fc_output = self.fully_connected(pool_output)
        print(f"Step 5 - After Fully Connected Layer: Flattened and passed through fully connected layer. Output Shape: {fc_output.shape}")
        print(f"Output Values (First 5): {fc_output[:5]}\n")
        
        # Step 6: Softmax Layer
        softmax_output = self.softmax(fc_output)
        print(f"Step 6 - After Softmax: Converted scores to class probabilities. Output Shape: {softmax_output.shape}")
        print(f"Class Probabilities: {softmax_output}\n")
        
        return softmax_output

# Example run with random 28x28 image
image = np.random.randn(28, 28)
cnn = SimpleCNN()
output = cnn.forward(image)


-- Practical_9 --

#pip install numpy scikit-fuzzy matplotlib

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Define universe of discourse
temp_current = np.arange(0, 41, 1)
temp_desired = np.arange(0, 41, 1)
heater_power = np.arange(0, 101, 1)

# Define membership functions
current_mfs = [
    fuzz.trapmf(temp_current, [0, 0, 10, 15]),  # Cold
    fuzz.trimf(temp_current, [10, 20, 30]),     # Warm
    fuzz.trapmf(temp_current, [25, 30, 40, 40]) # Hot
]
desired_mfs = [
    fuzz.trapmf(temp_desired, [0, 0, 10, 15]),  # Low
    fuzz.trimf(temp_desired, [10, 20, 30]),      # Medium
    fuzz.trapmf(temp_desired, [25, 30, 40, 40])  # High
]
power_mfs = [
    fuzz.trapmf(heater_power, [0, 0, 10, 20]),   # Off
    fuzz.trimf(heater_power, [10, 30, 50]),      # Low
    fuzz.trimf(heater_power, [30, 50, 70]),      # Medium
    fuzz.trapmf(heater_power, [50, 70, 90, 100]) # High
]

# Plot membership functions
fig, ax = plt.subplots(3, figsize=(10, 8))
titles = ['Current Temperature', 'Desired Temperature', 'Heater Power']
for i, (mfs, title) in enumerate(zip([current_mfs, desired_mfs, power_mfs], titles)):
    for mf in mfs:
        ax[i].plot(np.arange(len(mf)), mf)
    ax[i].set_title(title)
    ax[i].set_xlabel('Temperature (°C)' if i < 2 else 'Power (%)')
    ax[i].set_ylabel('Membership degree')

plt.tight_layout()
plt.show()

# Input values
current_temp, desired_temp = 18, 25
current_levels = [fuzz.interp_membership(temp_current, mf, current_temp) for mf in current_mfs]
desired_levels = [fuzz.interp_membership(temp_desired, mf, desired_temp) for mf in desired_mfs]

# Apply rules and aggregate
power_activation = np.fmax(
    np.fmax(current_levels[0] * desired_levels[0], 0) * power_mfs[0],  # Rule 1
    np.fmax(current_levels[1] * desired_levels[1], 0) * power_mfs[1],  # Rule 2
    np.fmax(current_levels[2] * desired_levels[2], 0) * power_mfs[3]   # Rule 3
)

# Defuzzify
output_power = fuzz.defuzz(heater_power, power_activation, 'centroid')
print(f"Recommended heater power: {output_power:.2f}%")
