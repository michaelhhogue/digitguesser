import pickle
import os
import re
from mytorch import nn
from mytorch.engine import Value
from digitdraw import Window

# Get the first generated model
MODEL_PATH = f"{os.getcwd()}/digit-guesser-model.pkl"

model = nn.MLP(100, [33, 33, 10])
model.load_model(MODEL_PATH)

window = Window()

def evaluation_action(grid_data):
    outputs = model(grid_data)
    
    # Search for highest probability output
    highest_i = 0
    highest_p = 0
    for i in range(len(outputs)):
        print(i, outputs[i])
        if outputs[i].data > highest_p:
            highest_i = i
            highest_p = outputs[i].data

    window.set_prediction_label(highest_i) 

# Set the action to call when the "Guess digit"
# button is pressed.
window.set_evaluation_action(evaluation_action)

# Show the GUI
window.show()
