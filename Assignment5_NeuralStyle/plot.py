import matplotlib.pyplot as plt
y = [848769540.0,610440060.0,425634750.0,318324480.0,256442140.0,218572260.0,193995100.0,176284780.0,162284910.0,150438430.0,140253630.0,131343740.0, 123392900.0] 
x = [i*20 for i,z in enumerate(y)]
plt.plot(x,y,"g-")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig("training_curve.jpg")
plt.show()