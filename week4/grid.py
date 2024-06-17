import matplotlib.pyplot as plt
#dataset
x=[1,2,3,4]
y=[6,7,8,9]


plt.title("Line plot with grid")
plt.xlabel("x-axis")
plt.ylabel("y-axis")

plt.plot(x, y,linestyle=':',color='m', markersize='7',marker='*',markerfacecolor='b')

plt.grid()
plt.legend()
plt.show()