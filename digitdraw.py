import tkinter as tk
from mytorch.engine import Value

class Window:

    def __init__(self, training_mode=False):
        # Grid parameters
        self._GRID_SIZE = 10   # The number of squares in the grid (in both dimensions)
        self._SQUARE_SIZE = 50  # The size of each square in the grid (in pixels)

        # Initialize the grid data
        self._grid_data = [0 for _ in range(self._GRID_SIZE * self._GRID_SIZE)]

        # Create the Tkinter window and canvas
        self._root = tk.Tk()
        self._root.title(f"Digit Guesser {'[Trainer]' if training_mode else ''}")

        self._canvas = tk.Canvas(
                self._root, 
                width=self._GRID_SIZE*self._SQUARE_SIZE, 
                height=self._GRID_SIZE*self._SQUARE_SIZE,
                background="white")

        self._canvas.pack()

        # Bind the fill_square function to mouse click and mouse motion events
        self._canvas.bind("<B1-Motion>", self._fill_square)
        self._canvas.bind("<Button-1>", self._fill_square)

        clear_btn = tk.Button(self._root, text="Clear", command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT)

        # Add the footer for either training or evaluation mode
        if training_mode:
            self._add_training_footer()
        else:
            self._add_evaluation_footer()

    def _add_training_footer(self):
        tk.Label(self._root, text="Intended Value:").pack(side=tk.LEFT)

        self._intended_res_input = tk.Entry(self._root)
        self._intended_res_input.pack(side=tk.LEFT)

        self._add_training_data_btn = tk.Button(
                self._root,
                text="Add to training data")

        self._train_btn = tk.Button(
                self._root,
                text="Train Model"
                )

        self._add_training_data_btn.pack(side=tk.LEFT)
        self._train_btn.pack(side=tk.LEFT)

    def _add_evaluation_footer(self):
        self._evaluate_btn = tk.Button(
                self._root,
                text="Guess digit")

        self._prediction_label = tk.Label(
                self._root,
                text="Prediction: -")

        self._evaluate_btn.pack(side=tk.LEFT)
        self._prediction_label.pack()

    def _draw_square(self, x, y, color="black"):
        self._canvas.create_rectangle(
                x,
                y,
                x + self._SQUARE_SIZE,
                y + self._SQUARE_SIZE,
                fill=color,
                outline="")

    def _fill_square(self, event):
        grid_x, grid_y = event.x // self._SQUARE_SIZE, event.y // self._SQUARE_SIZE

        index = grid_y * self._GRID_SIZE + grid_x
        if self._grid_data[index] == 0:
            self._grid_data[index] = 1
            self._draw_square(grid_x * self._SQUARE_SIZE, grid_y * self._SQUARE_SIZE)

    def clear_canvas(self):
        self._canvas.delete("all")
        self._grid_data = [0 for _ in range(self._GRID_SIZE * self._GRID_SIZE)]

    def show(self):
        # Start the Tkinter event loop
        self._root.mainloop()

    def set_training_data_action(self, action):
        self._add_training_data_btn.config(
                command=lambda: action(
                    self._grid_data,
                    self._intended_res_input.get()))

    def set_training_action(self, action):
        self._train_btn.config(command=action)

    def set_evaluation_action(self, action):
        self._evaluate_btn.config(command=lambda: action(self._grid_data))

    def set_prediction_label(self, prediction):
        self._prediction_label.config(text=f"Prediction: {prediction}")
