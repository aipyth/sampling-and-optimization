import numpy as np
import matplotlib.pyplot as plt

def gradient(f, x, eps=10-5):
    """
    Computes the gradient using finite difference scheme
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_forward = np.copy(x)
        x_backward = np.copy(x)
        x_forward[i] += eps
        x_backward[i] -= eps
        grad[i] = (f(x_forward) - f(x_backward)) / (2 * eps)
    return grad


def plot_gradient_descent(f, history, history_acc=None, bounds=None, title='Gradient Descent'):
    """
    Plots the results of a gradient descent algorithm.
    
    Args:
        f: The objective function to minimize.
        history: Array of points [x1, x2, ...] representing iteration points (for vanilla gradient descent).
        history_acc: Optional array of points representing iteration points for accelerated gradient descent (or another algorithm).
        bounds: Optional, tuple (min, max) bounds for plotting the contour of the function (used in 2D).
        title: Title of the plot.
    """
    # Check if the points are 1D or 2D
    history = np.array(history)
    
    # Plot 1D case
    if history.shape[1] == 1:
        plt.figure(figsize=(8, 6))
        
        # Create a range of x values for plotting the function
        x_vals = np.linspace(bounds[0], bounds[1], 400) if bounds else np.linspace(-5, 5, 400)
        y_vals = f(x_vals)
        
        # Plot the objective function
        plt.plot(x_vals, y_vals, label='Objective Function', color='blue')
        
        # Plot the points from the history of gradient descent
        points = np.array(history)
        plt.plot(points, f(points), 'ro-', label='Gradient Descent Iterations')
        
        if history_acc is not None:
            points_acc = np.array(history_acc)
            plt.plot(points_acc, f(points_acc), 'go-', label='Accelerated GD Iterations')
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()
    
    # Plot 2D case (for functions of two variables)
    elif history.shape[1] == 2:
        plt.figure(figsize=(10, 8))
        
        # Create a grid of points over the bounds to plot the contours
        if bounds:
            x_vals = np.linspace(bounds[0], bounds[1], 400)
            y_vals = np.linspace(bounds[0], bounds[1], 400)
        else:
            x_vals = np.linspace(-5, 5, 400)
            y_vals = np.linspace(-5, 5, 400)
        
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = np.array([[f([x, y]) for x in x_vals] for y in y_vals])
        
        # Plot the contour of the function
        plt.contour(X, Y, Z, levels=90, cmap='viridis')
        # plt.imshow()
        
        # Plot the points from the history of gradient descent
        points = np.array(history)
        # plt.plot(points[:, 0], points[:, 1], 'ro-', markersize=4, linewidth=1, label='Gradient Descent Iterations')
        
        if history_acc is not None:
            # points_acc = np.array(history_acc)
            plt.plot(history_acc[:, 0, 0], history_acc[:, 0, 1], 'go-', markersize=4, linewidth=1, label='Accelerated GD Iterations')
            plt.plot(history_acc[:, 0, 1], history_acc[:, 1, 1], 'bo-', markersize=4, linewidth=1, label='Accelerated GD Iterations')
        
        plt.plot(points[:, 0], points[:, 1], 'ro-', markersize=4, linewidth=1, label='Gradient Descent Iterations')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()
    else:
        raise ValueError("This plot function only supports 1D or 2D optimization problems.")
