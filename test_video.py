
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas

width = 800
height = 800
totalFrames = 10
out = cv.VideoWriter("test.mp4", cv.VideoWriter_fourcc(*'MP4V'),30.0,(width,height))
for frameNo in range(totalFrames):				
	fig = plt.figure(figsize=(8, 8)) # ,constrained_layout=True,dpi=100
	canvas = FigureCanvas(fig)

	ax = fig.add_subplot()
	ax.plot([5.5, 2.1], [1.4, 3.2])
	canvas.draw()
	plt.close()

	X = np.array(canvas.renderer.buffer_rgba())
	new_X = np.delete(X.reshape(-1,4),3,1)
	new_X = new_X.reshape(X.shape[0],X.shape[1],-1)
	print(new_X.shape)
	out.write(new_X)
	plt.imshow(new_X)
	plt.close()
	# break

out.release()
cv.destroyAllWindows()