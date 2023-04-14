import numpy as np
import matplotlib.pyplot as plt
from regression import multi_regress

def main():

    input_data = np.loadtxt("C:\\Users\\mcdeb\\GOPH420\\GOPH420-W2023-LAB03\\data\\M_data.txt")
    time = (input_data[:,0])
    mag = (input_data[:,1])
    colors = ["r","b","g","y","c"]

    plt.figure()
    plt.xlabel("Time [h]")
    plt.ylabel("Magnitude [M]")
    plt.title("Earthquake Event Data")
    plt.plot(time,mag,"c.")
    plt.grid()
    plt.savefig('C:\\Users\\mcdeb\\GOPH420\\GOPH420-W2023-LAB03\\data\\data.png')

    boundaries = [1e-8,34,45,71,95,time[-1]]
    incr = np.array([-0.5,-0.25,0,0.25,0.5,0.75,1.0])
    
    k = 0
    index = np.zeros(0)
    for i in boundaries:
        while i > time[k]:
            k = k + 1
        index = np.append(index,k)

    def count(boundaries, interval):
        n = np.zeros(0)
        for i in range(len(boundaries)):
            n = np.append(n,np.sum(np.where(interval>=boundaries[i],1,0)))
        return n
    
    n_list = np.empty((5, len(incr)))
    for i in range (5):

        n = np.log10(count(incr,mag[int(index[i]):int(index[i+1])]))
        n_list[i,:] = n

    def model(n,m):

        y = np.transpose([n])
        z = np.transpose(np.array([np.ones(np.array(m).shape),m]))
        a,e,rsq,model = multi_regress(y,z)

        return a,e,rsq,model

    plt.figure()
    plt.title("log(N) as a function of M")
    plt.xlabel("Magnitude [M]")
    plt.ylabel("log(N) events")
    for i in range (5):
        plt.plot(incr,n_list[i],f"{colors[i]}.-",label = f"Event {i+1}")
    plt.legend()
    plt.grid()
    plt.savefig('C:\\Users\\mcdeb\\GOPH420\\GOPH420-W2023-LAB03\\data\\events.png')

    plt.style.use('ggplot')
    fig, axs = plt.subplots(5, 1, figsize=(8, 10))
    fig.subplots_adjust(hspace=0.6)
    for i in range(5):
        axs[i].plot(incr, n_list[i], f"{colors[i]}o", label=f"Event {i+1}")
        a, e, rsq, mod = model(n_list[i], incr)
        axs[i].plot(incr, mod, "k--")
        axs[i].set_title(f"Event {i+1}: A={a}, RSQ={(float(rsq)):.2f}", loc='left', fontsize=10)
        axs[i].set_xlabel("Magnitude [M]", fontsize=8)
        axs[i].set_ylabel("log(N) events", fontsize=8)
        axs[i].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('C:\\Users\\mcdeb\\GOPH420\\GOPH420-W2023-LAB03\\data\\models.png')

    plt.show()

if __name__ == "__main__":
        main()