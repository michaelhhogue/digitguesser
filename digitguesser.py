import pickle
import os
from digitdraw import Window
from neural_network import NeuralNetwork
import torch

model_file_name = input("Enter the model file name: ")

if not os.path.isfile(model_file_name):
    print(f"Could not find a model file named {model_file_name}")
    exit()

model = NeuralNetwork(False)
model.load_state_dict(torch.load(model_file_name))

window = Window()

def evaluation_action(grid_data):
    x = normalize_grid_data(grid_data)

    model.eval()
    with torch.no_grad():
        output = model(x)

        window.set_prediction_label(output.argmax()) 


def normalize_grid_data(grid_data):
    t = torch.tensor(grid_data, dtype=torch.float)
    mean, std = torch.mean(t), torch.std(t)
    return (t - mean) / std

# Set the action to call when the "Guess digit"
# button is pressed.
window.set_evaluation_action(evaluation_action)

# Show the GUI
window.show()
