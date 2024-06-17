import matplotlib.pyplot as plt

#Data
x = [1,2,3,4,5]
y = [2,3,6,7,10]

plt.plot(x,y,marker='o',linestyle='--',color='b',markersize=6,markerfacecolor='r')
plt.title("line plot with markers")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()


plt.plot(x,y,marker='.',linestyle='--',color='b',markersize=6,markerfacecolor='r')
plt.title("line plot with markers")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()