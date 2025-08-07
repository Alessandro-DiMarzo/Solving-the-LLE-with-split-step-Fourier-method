import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq, fftshift
import matplotlib.animation as animation

class LLE_sim:
    def __init__(self, eta, detuning, F, t_R, roundtrips, N):
        self.eta = eta
        self.detuning = detuning
        self.F = F
        self.t_R = t_R
        self.roundtrips = roundtrips
        self.N = N

        # Fast time grid (over 1 round trip)
        self.tau = np.linspace(-self.t_R/2, self.t_R/2, self.N)
        self.dtau = self.tau[1] - self.tau[0]

        # Frequency grid
        self.omega = 2 * np.pi * fftfreq(self.N, d=self.dtau)

        # Save evolution
        self.E_evolution = np.zeros((self.roundtrips, self.N), dtype=complex)
        self.t_vals = np.arange(self.roundtrips)
        
    def create_pulse(self, height, std):       #for initial E
        return height * np.exp(-(self.tau)**2/(std))
    
    def nonlinear_step(self, A):  
        return (- 1 - 1j * self.detuning + 1j  * np.abs(A)**2) * A
    
    def ssfm(self): #solve LLE
        dt = 0.01
        E = 0.75 + 1.8 * np.exp(-self.tau**2 / (0.75)) #initial pulse
        D = np.exp(1j * self.eta * self.omega**2 * dt)
        for step in range(self.roundtrips):
            #linear step 
            E = fft(E)
            E = ifft(E * D)
            
            #nl step with detuning, loss, driving (using rk2)
            k1 = dt * (self.nonlinear_step(E) + self.F)
            k2 = dt * (self.nonlinear_step(E + k1/2) +self.F)
            E = (E + k2)
            self.E_evolution[step] = E #save intensity profile for slow time plots
        return E

    def plot_peak_power(self):

        peak_power = np.max(np.abs(self.E_evolution)**2, axis=1)
        plt.figure()
        plt.title('Peak Power of E')
        plt.plot(self.t_vals, peak_power)
        plt.xlabel('Slow time t (s)')
        plt.ylabel('W')
        plt.grid()
        
    def get_spectrum(self, sol):
        spectrum = np.fft.fft(sol)
        frequencies = np.fft.fftfreq(len(sol), self.dtau)
        spectrum_magnitude = np.abs(np.fft.fftshift(spectrum))
        frequencies_shifted = np.fft.fftshift(frequencies)
        return frequencies_shifted, spectrum_magnitude
        
    def plot_slow_time_evolution(self):
        self.ssfm()
        if self.eta == -1:
            print("η =", self.eta, " (Anomalous dispersion)")
        elif self.eta == 1:
            print("η =", self.eta, " (Normal dispersion)")
            
        print("detuning =", self.detuning)
        print("F =", self.F)

        
        # Plot space-time diagram of intensity
        plt.figure()
        plt.imshow(np.abs(self.E_evolution)**2, aspect='auto', extent=[self.tau[0], self.tau[-1], self.t_vals[0], self.t_vals[-1]], cmap='jet' , origin='lower')
        plt.xlabel('Fast time τ')
        plt.ylabel('Slow time t (roundtrips)')
        plt.title('Evolution of |E(t, τ)|²')
        plt.colorbar(label='|E(t, τ)|²')
        plt.show()
        
        # animate the above plot
        fig, axs = plt.subplots(2, 1)
        fast_time = np.linspace(-self.t_R/2, self.t_R/2, self.N)
        line, = axs[0].plot([], [], lw=2)
        
        axs[0].set_xlim(fast_time[0], fast_time[-1])
        axs[0].set_ylim(0, np.max(np.abs(self.E_evolution)**2) * 1.5)
        axs[0].set_xlabel('Fast Time')
        axs[0].set_ylabel('Intensity |E|^2 ')
    
        def init():
            line.set_data([], [])
            return line,
    
        def update(frame):
            axs[0].set_ylim(0, np.max(np.abs(self.E_evolution[frame])**2) * 1.5)
            line.set_data(fast_time, np.abs(self.E_evolution[frame])**2)
            axs[0].set_title(f'roundtrips: {frame}', y = 1.0, pad=-14)
            
            return line,
    
        self.ani = animation.FuncAnimation(fig, update, frames=len(self.E_evolution), init_func=init, blit=False, interval=75)
        
        total_power = []
        
        for i in range(len(self.t_vals)):
            total_power.append(sum(np.abs(self.E_evolution[i])**2))
        
        
        axs[1].plot(self.t_vals, total_power)
        axs[1].set(xlabel = 'Slow time t (roundtrips)', ylabel = 'Total Power of E' )
        plt.tight_layout()
        
        self.ani.save('dimless_slow_time_evo.gif', writer = 'pillow')
        
    def scan_detuning(self, delta_start, delta_end, delta_steps): #as if we chnage detuning every 1000 roundtrips
        delta_values = np.linspace(delta_start,delta_end, delta_steps)
        E_profiles = []
        for delta in delta_values:
            self.detuning = delta
            E_profiles.append(np.abs(self.ssfm())**2)
        
        #create plot of intracavity power and intensity profile as we scan detuning
        fig, axs = plt.subplots(3, 1)
        fast_time = np.linspace(-self.t_R/2, self.t_R/2, self.N)
        line, = axs[0].plot([], [], lw=2)
        line1, = axs[1].plot([], [], lw=2)
        
        axs[0].set_xlim(fast_time[0], fast_time[-1])
        axs[0].set_ylim(0, np.max(E_profiles) * 1.5)
        axs[0].set_xlabel('Fast Time')
        axs[0].set_ylabel('Intensity |E|^2')
        axs[1].set_xlim(self.get_spectrum(E_profiles)[0][1], self.get_spectrum(E_profiles)[0][-1])
        axs[1].set_xlabel('Frequency')
        axs[1].set_ylabel('Spectrum')
        
        def init():
            line.set_data([], [])
            line1.set_data([], [])
            return line,line1
    
        def update(frame):
            line.set_data(fast_time, (E_profiles[frame]))
            line1.set_data(self.get_spectrum(E_profiles[frame])[0],self.get_spectrum(E_profiles[frame])[1] )
            axs[1].set_xlim(np.min(self.get_spectrum(E_profiles[frame])[0]), np.max(self.get_spectrum(E_profiles[frame])[0]))
            axs[1].set_ylim(0, np.max(self.get_spectrum(E_profiles[frame])[1]) * 1.5)
            axs[0].set_ylim(0, np.max(E_profiles[frame]) * 1.5)
            axs[0].set_title(f'normalized detuning: {round(delta_values[frame],2)}', y = 1.0, pad=-14)
            return line,line1
    
        self.ani = animation.FuncAnimation(fig, update, frames=len(E_profiles), init_func=init, blit=False, interval=100)
        total_power = []
        
        
        for i in range(len(delta_values)):
            total_power.append(sum((E_profiles[i])))
        
        
        axs[2].plot(delta_values, total_power)
        axs[2].set(xlabel = 'normalized detuning', ylabel = 'Total Power of E' )
        plt.tight_layout()
        self.ani.save('dimless_detuning_evo.gif', writer = 'pillow')
        
        # Fast time intensity profile
        plt.figure()
        plt.title('Fast Time Intensity Profile of |E(t, τ)|² as we Change Detuning')
        plt.imshow(E_profiles, extent=[self.tau[0], self.tau[-1], delta_values[0], delta_values[-1]], aspect='auto', cmap='hot', origin='lower')
        plt.xlabel('Fast time τ')
        plt.ylabel('Detuning (Hz)')
        plt.colorbar(label = '|E(t, τ)|²')
        plt.tight_layout()
        plt.show()    
        
sol = LLE_sim(         
    eta= -1,                   
    detuning= 1.5,     
    F = np.sqrt(2),              
    t_R= 20,           
    roundtrips=1000,
    N=2**9
)

#%%
sol.scan_detuning(-0.15,2.5,200)
#%%
sol.plot_slow_time_evolution()