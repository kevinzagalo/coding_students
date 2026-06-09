import matplotlib.pyplot as plt


class InteractiveGrid:
    def __init__(self, grid):
        self.grid = grid
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.grid, cmap='binary')
        self.ax.set_title('Click on the pixels to alter them.')
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()

    def on_click(self, event):
        if event.inaxes is not None:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]:
                self.grid[y, x] = 1 - self.grid[y, x]  # toggle color
                self.ax.imshow(self.grid, cmap='binary')
                plt.draw()