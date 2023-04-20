import numpy as np
import matplotlib.pyplot as plt
from regression import multi_regress

def main():

    #Reads input file and creates lists of timestamps and magnitudes. Creates a list of plot colors.
    input_data = np.loadtxt("C:\\Users\\mcdeb\\GOPH420\\GOPH420-W2023-LAB03\\data\\M_data.txt")
    time = (input_data[:,0])
    mag = (input_data[:,1])
    colors = ["r","b","g","y","c"]

    #Plots and saves input data.
    plt.figure()
    plt.xlabel("Time [h]")
    plt.ylabel("Magnitude [M]")
    plt.title("Figure 1: Earthquake Data")
    plt.plot(time,mag,"c.")
    plt.grid()
    plt.savefig('C:\\Users\\mcdeb\\GOPH420\\GOPH420-W2023-LAB03\\data\\data.png')

    #5 distinct events are evident from fig. 1: Creates boundaries for these events and plotting increments for later figures.
    boundaries = [1e-8,34,45,71,96,time[-1]]
    incr = np.array([-1.0,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.0])

    #Plots vertical lines indicating event boundaries
    plt.figure()
    plt.xlabel("Time [h]")
    plt.ylabel("Magnitude [M]")
    plt.title("Figure 2: Boundary Data")
    plt.plot(time,mag,"c.")
    plt.vlines(boundaries[1:5],-1.5,2)
    plt.grid()
    plt.savefig('C:\\Users\\mcdeb\\GOPH420\\GOPH420-W2023-LAB03\\data\\boundaries.png')
    
    #Creates a list of indices in time where boundary cutoffs occur
    k = 0
    index = np.zeros(0)
    for i in boundaries:
        while i > time[k]:
            k = k + 1
        index = np.append(index,k)

    #Counts the number of events of each magnitude in incr that occur within each boundary set
    def count(slices, window):
        n = np.zeros(0)
        for i in range(len(slices)):
            n = np.append(n,np.sum(np.where(window>=slices[i],1,0)))
        return n
    
    #Creates lists of log(n) events that occur at or above each magnitude specified in incr
    n_list = np.empty((5, len(incr)))
    for i in range (5):

        n = np.log10(count(incr,mag[int(index[i]):int(index[i+1])]))
        n_list[i,:] = n

    #Creates arrays for function outputs and variables, and passes them to model. defines model parameters, residuals, r^2, and output.
    def model(n,m):

        y = np.transpose([n])
        z = np.transpose(np.array([np.ones(np.array(m).shape),m]))
        a,e,rsq,model = multi_regress(y,z)

        return a,e,rsq,model

    #Plots log(n) events for each magnitude slice as a function of event magnitude based on raw data
    plt.figure()
    plt.title("Figure 3: log(N) as a function of M")
    plt.xlabel("Magnitude [M]")
    plt.ylabel("log(N) events")
    for i in range (5):
        plt.plot(incr,n_list[i],f"{colors[i]}.-",label = f"Event {i+1}")
    plt.legend()
    plt.grid()
    plt.savefig('C:\\Users\\mcdeb\\GOPH420\\GOPH420-W2023-LAB03\\data\\events.png')

    #Creates a figure with 5 subplots plotting log(n) events as a function of magnitude based on model output
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
        axs[i].grid()
    plt.tight_layout()
    plt.savefig('C:\\Users\\mcdeb\\GOPH420\\GOPH420-W2023-LAB03\\data\\models.png')

    #Displays the plots
    plt.show()

#Runs the program
if __name__ == "__main__":
        main()