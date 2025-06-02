import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
import math
class SignalGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("DSP Framework")
        self.root.geometry("1100x700")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both")

        self.create_generator_page()
        self.create_arithmetic_page()
        self.create_quantization_page()
        self.create_transform_page()
        self.create_td_page()
        self.create_six_page()

        self.signals = []  # Store loaded signals
        self.current_signal = None

    def create_generator_page(self):
        generator_frame = ttk.Frame(self.notebook)
        self.notebook.add(generator_frame, text="Signal Generator")

        # Title Label
        ttk.Label(generator_frame, text="Signal Generator", font=('Arial', 16, 'bold')).grid(column=0, row=0,
                                                                                             columnspan=3, pady=(0, 15))

        # Amplitude
        ttk.Label(generator_frame, text="Amplitude (A):", font=('Arial', 12)).grid(column=0, row=1, sticky=tk.W, pady=5)
        self.amplitude_entry = ttk.Entry(generator_frame, width=30)
        self.amplitude_entry.grid(column=1, row=1, sticky=(tk.W, tk.E))

        # Analog Frequency
        ttk.Label(generator_frame, text="Analog Frequency (Hz):", font=('Arial', 12)).grid(column=0, row=2, sticky=tk.W,
                                                                                           pady=5)
        self.frequency_entry = ttk.Entry(generator_frame, width=30)
        self.frequency_entry.grid(column=1, row=2, sticky=(tk.W, tk.E))

        # Sampling Frequency
        ttk.Label(generator_frame, text="Sampling Frequency (Hz):", font=('Arial', 12)).grid(column=0, row=3,
                                                                                             sticky=tk.W, pady=5)
        self.sampling_entry = ttk.Entry(generator_frame, width=30)
        self.sampling_entry.grid(column=1, row=3, sticky=(tk.W, tk.E))

        # Phase Shift
        ttk.Label(generator_frame, text="Phase Shift (radians):", font=('Arial', 12)).grid(column=0, row=4, sticky=tk.W,
                                                                                           pady=5)
        self.phase_entry = ttk.Entry(generator_frame, width=30)
        self.phase_entry.grid(column=1, row=4, sticky=(tk.W, tk.E))

        # Signal Type
        ttk.Label(generator_frame, text="Select Signal Type:", font=('Arial', 12)).grid(column=0, row=5, sticky=tk.W,
                                                                                        pady=5)
        self.signal_type_var = tk.StringVar(value="cos")  # Default value
        ttk.Radiobutton(generator_frame, text="Sinusoidal", variable=self.signal_type_var, value="sin").grid(column=0,
                                                                                                             row=6,
                                                                                                             sticky=tk.W,
                                                                                                             padx=(
                                                                                                             20, 0))
        ttk.Radiobutton(generator_frame, text="Cosinusoidal", variable=self.signal_type_var, value="cos").grid(column=1,
                                                                                                               row=6,
                                                                                                               sticky=tk.W)

        # Generate Button
        generate_button = ttk.Button(generator_frame, text="Generate Wave", command=self.generate_wave)
        generate_button.grid(column=0, row=7, columnspan=2, pady=20)

        # Additional Styling (Optional)
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="lightblue")
        style.configure("TLabel", padding=(10, 5))
        style.configure("TEntry", padding=5)

    def generate_wave(self):
        # Get values from entry fields
        signal_type = self.signal_type_var.get()
        A = float(self.amplitude_entry.get())
        AnalogFrequency = float(self.frequency_entry.get())
        SamplingFrequency = float(self.sampling_entry.get())
        PhaseShift = float(self.phase_entry.get())

        # Generate time values based on sampling frequency
        t = np.arange(0, int(SamplingFrequency), 1)

        # Generate the signal based on the type
        if signal_type == 'sin':
            samples = A * np.sin(2 * np.pi * AnalogFrequency / int(SamplingFrequency) * t + PhaseShift)
        elif signal_type == 'cos':
            samples = A * np.cos(2 * np.pi * AnalogFrequency / int(SamplingFrequency) * t + PhaseShift)
        else:
            raise ValueError("Unsupltorted signal type. Use 'sin' or 'cos'.")

        # Plotting the generated signal
        plt.figure(figsize=(10, 4))
        plt.plot(t, samples)
        plt.title(f'{signal_type.capitalize()} Wave')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.xlim(0, 0.04)
        plt.show()

        self.SignalSamplesAreEqual("CosOutput.txt", t, samples)
        self.save_output_file("Saved.txt", t, samples)

    def create_arithmetic_page(self):
        arithmetic_frame = ttk.Frame(self.notebook)
        self.notebook.add(arithmetic_frame, text="Arithmetic Operations")

        # File selection
        ttk.Label(arithmetic_frame, text="Select Signal Files:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
        self.file_listbox = tk.Listbox(arithmetic_frame, width=50, height=5)
        self.file_listbox.grid(row=1, column=0, columnspan=2, padx=10, pady=5)

        #ttk.Button(arithmetic_frame, text="Add File", command=self.add_file).grid(row=2, column=0, sticky="w", padx=10, pady=5)
        ttk.Button(arithmetic_frame, text="Remove File", command=self.remove_file).grid(row=2, column=1, sticky="w", padx=10, pady=5)

        # Operation selection
        ttk.Label(arithmetic_frame, text="Select Operation:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
        self.operation_var = tk.StringVar()
        operations = ["Addition", "Subtraction", "Multiplication", "Squaring", "Normalization", "Accumulation"]
        self.operation_combobox = ttk.Combobox(arithmetic_frame, textvariable=self.operation_var, values=operations)
        self.operation_combobox.grid(row=3, column=1, sticky="w", padx=10, pady=5)
        self.operation_combobox.set(operations[0])

        # Constant for multiplication
        self.constant_var = tk.DoubleVar(value=1.0)
        self.constant_entry = ttk.Entry(arithmetic_frame, textvariable=self.constant_var, width=10)
        self.constant_entry.grid(row=4, column=1, sticky="w", padx=10, pady=5)
        ttk.Label(arithmetic_frame, text="Constant (for multiplication):").grid(row=4, column=0, sticky="w", padx=10, pady=5)

        # Execute button
        ttk.Button(arithmetic_frame, text="Execute Operation", command=self.execute_operation).grid(row=5, column=0, columnspan=2, pady=20)
   
    def create_transform_page(self):
        transform_frame = ttk.Frame(self.notebook)
        self.notebook.add(transform_frame, text="Frequency Domain")
        file_frame = ttk.LabelFrame(transform_frame, text="File Selection", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(file_frame, text="Input File:").grid(row=0, column=0, sticky="w", pady=5)
        self.fd_file_path_entry = ttk.Entry(file_frame, width=50)
        self.fd_file_path_entry.grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.select_file(self.fd_file_path_entry)).grid(row=0,column=2)

        ttk.Label(file_frame, text="Test File:").grid(row=1, column=0, sticky="w", pady=5)
        self.fd_test_file_entry = ttk.Entry(file_frame, width=50)
        self.fd_test_file_entry.grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.select_file(self.fd_test_file_entry)).grid(row=1, column=2)

        fs_frame = ttk.LabelFrame(transform_frame, text="Sampling Frequency", padding=10)
        fs_frame.pack(fill="x", padx=10, pady=5)

        self.fs_entry = ttk.Entry(fs_frame, width=20)
        self.fs_entry.grid(row=1, column=1, columnspan=2, sticky="w")
    
        dct_frame = ttk.LabelFrame(transform_frame, text="DCT", padding=10)
        dct_frame.pack(fill="x", padx=10, pady=5)

        fs_frame = ttk.LabelFrame(transform_frame, text="No. of Coefficients", padding=10)
        fs_frame.pack(fill="x", padx=10, pady=5)

        self.coeff_entry = ttk.Entry(fs_frame, width=20)
        self.coeff_entry.grid(row=1, column=1, columnspan=2, sticky="w")

        ttk.Button(transform_frame, text="Calculate DCT", command=self.DCT).pack(pady=20)

        ttk.Button(transform_frame, text="Run Transformation/Reconstruction", command=self.run_frequency_domain).pack(pady=20)

    def create_td_page(self):
        td_frame = ttk.Frame(self.notebook)
        self.notebook.add(td_frame, text="Time Domain")
        
        # File Selection Frame
        file_frame = ttk.LabelFrame(td_frame, text="File Selection", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(file_frame, text="Input File:").grid(row=0, column=0, sticky="w", pady=5)
        self.td_file_path_entry = ttk.Entry(file_frame, width=50)
        self.td_file_path_entry.grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.select_file(self.td_file_path_entry)).grid(row=0, column=2)
        
        ttk.Label(file_frame, text="Test File:").grid(row=1, column=0, sticky="w", pady=5)
        self.td_test_file_entry = ttk.Entry(file_frame, width=50)
        self.td_test_file_entry.grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.select_file(self.td_test_file_entry)).grid(row=1, column=2)
        
        fs_frame = ttk.LabelFrame(td_frame, text="Steps", padding=10)
        fs_frame.pack(fill="x", padx=10, pady=5)
        
        self.steps_entry = ttk.Entry(fs_frame, width=20)
        self.steps_entry.grid(row=1, column=1, columnspan=2, sticky="w")

        filter_frame = ttk.LabelFrame(td_frame, text="Filter Parameters", padding=10)
        filter_frame.pack(side="left", padx=10, pady=5)

        ttk.Label(filter_frame, text="Filter Type:").grid(row=0, column=0, sticky="w", pady=5)
        self.filter_type_menu = ttk.Combobox(filter_frame, values=["Low Pass", "High Pass", "Band Pass", "Band Stop"], state="readonly", width=20)
        self.filter_type_menu.grid(row=0, column=1, padx=5)
        
        ttk.Label(filter_frame, text="Sampling Frequency (Hz):").grid(row=1, column=0, sticky="w", pady=5)
        self.sampling_freq_entry = ttk.Entry(filter_frame, width=20)
        self.sampling_freq_entry.grid(row=1, column=1, padx=5)
        
        ttk.Label(filter_frame, text="Stop Band Attenuation (dB):").grid(row=2, column=0, sticky="w", pady=5)
        self.stop_atten_entry = ttk.Entry(filter_frame, width=20)
        self.stop_atten_entry.grid(row=2, column=1, padx=5)

        ttk.Label(filter_frame, text="Cutoff Frequency (Hz):").grid(row=3, column=0, sticky="w", pady=5)
        self.cutoff_freq_entry = ttk.Entry(filter_frame, width=20)
        self.cutoff_freq_entry.grid(row=3, column=1, padx=5)
        
        ttk.Label(filter_frame, text="Second Cutoff Frequency (Hz):").grid(row=4, column=0, sticky="w", pady=5)
        self.second_cutoff_freq_entry = ttk.Entry(filter_frame, width=20)
        self.second_cutoff_freq_entry.grid(row=4, column=1, padx=5)
        
        ttk.Label(filter_frame, text="Transition Band (Hz):").grid(row=5, column=0, sticky="w", pady=5)
        self.transition_band_entry = ttk.Entry(filter_frame, width=20)
        self.transition_band_entry.grid(row=5, column=1, padx=5)
        
        ttk.Label(filter_frame, text="Convolve?:").grid(row=6, column=0, sticky="w", pady=5)
        self.convOption = ttk.Combobox(filter_frame, values=["Yes","No"], state="readonly", width=20)
        self.convOption.grid(row=6, column=1, padx=5)

        resampling_frame = ttk.LabelFrame(td_frame, text="Resampling Parameters", padding=10)
        resampling_frame.pack(side="left", padx=10, pady=5)

        ttk.Label(resampling_frame, text="Upsampling Factor (L):").grid(row=0, column=0, sticky="w", pady=5)
        self.upsampling_factor_entry = ttk.Entry(resampling_frame, width=20)
        self.upsampling_factor_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(resampling_frame, text="Downsampling Factor (M):").grid(row=1, column=0, sticky="w", pady=5)
        self.downsampling_factor_entry = ttk.Entry(resampling_frame, width=20)
        self.downsampling_factor_entry.grid(row=1, column=1, padx=5)

        button_frame = ttk.Frame(td_frame, padding=10)
        button_frame.pack(side="left", fill="y", padx=10, pady=10)

        ttk.Button(button_frame, text="Derivative", command=self.DerivativeSignal).pack(padx=10, pady=10)
        ttk.Button(button_frame, text="Shift Only", command=self.shift_only).pack(padx=10, pady=10)
        ttk.Button(button_frame, text="Fold Only", command=self.fold_signal).pack(padx=10, pady=10)
        ttk.Button(button_frame, text="Shift Folded", command=self.shift_folded).pack(padx=10, pady=10)
        ttk.Button(button_frame, text="Filter", command=self.filtering).pack(padx=10, pady=10)
        ttk.Button(button_frame, text="Resample", command=self.resample).pack(padx=10, pady=10)

    def create_six_page(self):
        six_frame = ttk.Frame(self.notebook)
        self.notebook.add(six_frame, text="Lab Six")

        file_frame = ttk.LabelFrame(six_frame, text="File Selection", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(file_frame, text="Input File 1:").grid(row=0, column=0, sticky="w", pady=5)
        self.six_file_path_entry = ttk.Entry(file_frame, width=50)
        self.six_file_path_entry.grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.select_file(self.six_file_path_entry)).grid(row=0,column=2)

        ttk.Label(file_frame, text="Input File 2 (Convolution and Correlaion):").grid(row=1, column=0, sticky="w", pady=5)
        self.six_file_path_entry2 = ttk.Entry(file_frame, width=50)
        self.six_file_path_entry2.grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.select_file(self.six_file_path_entry2)).grid(row=1,column=2)

        ttk.Label(file_frame, text="Test File:").grid(row=2, column=0, sticky="w", pady=5)
        self.six_test_file_entry = ttk.Entry(file_frame, width=50)
        self.six_test_file_entry.grid(row=2, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.select_file(self.six_test_file_entry)).grid(row=2, column=2)

        window_frame = ttk.LabelFrame(six_frame, text="Window Size (Smoothing):", padding=10)
        window_frame.pack(fill="x", padx=10, pady=5)

        self.windowSizeEntry = ttk.Entry(window_frame, width=20)
        self.windowSizeEntry.grid(row=1, column=1, columnspan=2, sticky="w")

        ttk.Button(six_frame, text="Smoothing", command=self.smoothing).pack(pady=5)
        ttk.Button(six_frame, text="Remove DC Component (Time)", command=self.remove_dc_component_time).pack(pady=5)
        ttk.Button(six_frame, text="Remove DC Component (Frequency)", command=self.remove_dc_component_frequency).pack(pady=5)
        ttk.Button(six_frame, text="Convolve", command=self.convolution).pack(pady=5)
        ttk.Button(six_frame, text="Correlate", command=self.correlation).pack(pady=5)

    def filtering(self):
        file_name = self.td_file_path_entry.get()
        test_file = self.td_test_file_entry.get()

        filter_type = self.filter_type_menu.get() 
        sampling_freq = float(self.sampling_freq_entry.get())  
        cutoff_freq = float(self.cutoff_freq_entry.get()) 
        stop_atten = float(self.stop_atten_entry.get())
        transition_band = float(self.transition_band_entry.get()) 

        delta_f = transition_band / sampling_freq

        if 1 <= stop_atten <= 21:
            window_type = "Rectangular"
            window_factor = 0.9
        elif 22 <= stop_atten <= 44:
            window_type = "Hanning"
            window_factor = 3.1
        elif 45 <= stop_atten <= 53:
            window_type = "Hamming"
            window_factor = 3.3
        elif 54 <= stop_atten <= 74:
            window_type = "Blackman"
            window_factor = 5.5
        else:
            raise ValueError("Stop band attenuation out of range for known windows.")
        N = window_factor / delta_f
        N = math.ceil(N)  
        if N % 2 == 0:  
            N += 1
            
        n = np.arange(-(N // 2), N // 2 + 1)  
        hd = np.zeros_like(n, dtype=float)

        if filter_type == "Low Pass":
            fc_normalized = (cutoff_freq + (transition_band / 2)) / sampling_freq
            wc = 2 * np.pi * fc_normalized 
            hd[n == 0] = 2 * fc_normalized 
            hd[n != 0] = (2 * fc_normalized) * (np.sin(n[n != 0] * wc) / (n[n != 0] * wc))

        elif filter_type == "High Pass":
            fc_normalized = (cutoff_freq - (transition_band / 2)) / sampling_freq
            wc = 2 * np.pi * fc_normalized  
            hd[n == 0] = (1 - (2 * fc_normalized))
            hd[n != 0] = -2 * fc_normalized * (np.sin(n[n != 0] * wc) / (n[n != 0] * wc))

        elif filter_type == "Band Pass":
            f2 = float(self.second_cutoff_freq_entry.get())
            fc_normalized = (cutoff_freq - (transition_band / 2)) / sampling_freq
            f2_normalized = (f2 + (transition_band / 2)) / sampling_freq

            wc = 2 * np.pi * fc_normalized
            w2 = 2 * np.pi * f2_normalized

            hd[n == 0] = 2*(f2_normalized-fc_normalized)
            hd[n != 0] = ((2 * f2_normalized) * (np.sin(n[n != 0] * w2) / (n[n != 0] * w2))) - ((2 * fc_normalized) * (np.sin(n[n != 0] * wc) / (n[n != 0] * wc)))

        elif filter_type == "Band Stop":
            f2 = float(self.second_cutoff_freq_entry.get())

            fc_normalized = (cutoff_freq + (transition_band / 2)) / sampling_freq
            f2_normalized = (f2 - (transition_band / 2)) / sampling_freq

            wc = 2 * np.pi * fc_normalized
            w2 = 2 * np.pi * f2_normalized

            hd[n == 0] = 1 - 2 * (f2_normalized - fc_normalized)
            hd[n != 0] = ((2 * fc_normalized) * (np.sin(n[n != 0] * wc) / (n[n != 0] * wc))) - ((2 * f2_normalized) * (np.sin(n[n != 0] * w2) / (n[n != 0] * w2)))
        
        else:
            raise ValueError("Invalid filter type.")
        
        if window_type == "Rectangular":
            w = np.ones_like(n)
        elif window_type == "Hanning":
            w = 0.5 + 0.5 * np.cos(2 * np.pi * n / N)
        elif window_type == "Hamming":
            w = 0.54 + 0.46 * np.cos(2 * np.pi * n / N)
        elif window_type == "Blackman":
            w = 0.42 + 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))

        h = hd * w
        coefficients = [(int(idx), coeff) for idx, coeff in zip(n, h)]
        
        plt.figure(figsize=(8, 6))

        if self.convOption.get() == "No":
            print("Filter Coefficients (h[n]):")
            for idx, coeff in coefficients:
                print(f"{idx} {coeff:.9f}")

            Your_indices = [int(idx) for idx, _ in coefficients]
            Your_samples = [coeff for _, coeff in coefficients]
    
            self.Compare_Signals(test_file, Your_indices, Your_samples)

            plt.subplot(2, 1, 1)
            plt.plot(n, h, label=f'{filter_type} Filter Coefficients', color='green')
            plt.title(f'{filter_type} Filter Coefficients')
            plt.xlabel("Index")
            plt.ylabel("Amplitude")
            plt.grid(True)

        else:
            indices, samples = self.read_signal_file(file_name)
            Your_indices, Your_samples = self.convFilter(indices, samples, [idx for idx, _ in coefficients] ,[coeff for _, coeff in coefficients] )

            print("Filtered Signal:")
            for i in range(len(Your_indices)):
                print(f"{Your_indices[i]} {Your_samples[i]:.9f}")
            self.Compare_Signals(test_file, Your_indices, Your_samples)

            plt.subplot(2, 1, 1)
            plt.plot(indices, samples)
            plt.title("Original Signal")
            plt.xlabel("Index")
            plt.ylabel("Amplitude")

            plt.subplot(2, 1, 2)
            plt.plot(Your_indices, Your_samples)
            plt.title("Filtered Signal")
            plt.xlabel("Index")
            plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()
        return Your_indices, Your_samples
    
    def Filtering(self, indices, samples):

        sampling_freq = float(self.sampling_freq_entry.get())  
        cutoff_freq = float(self.cutoff_freq_entry.get()) 
        stop_atten = float(self.stop_atten_entry.get())
        transition_band = float(self.transition_band_entry.get()) 

        delta_f = transition_band / sampling_freq

        if 1 <= stop_atten <= 21:
            window_type = "Rectangular"
            window_factor = 0.9
        elif 22 <= stop_atten <= 44:
            window_type = "Hanning"
            window_factor = 3.1
        elif 45 <= stop_atten <= 53:
            window_type = "Hamming"
            window_factor = 3.3
        elif 54 <= stop_atten <= 74:
            window_type = "Blackman"
            window_factor = 5.5
        else:
            raise ValueError("Stop band attenuation out of range for known windows.")
        N = window_factor / delta_f
        N = math.ceil(N)  
        if N % 2 == 0:  
            N += 1
            
        n = np.arange(-(N // 2), N // 2 + 1)  
        hd = np.zeros_like(n, dtype=float)

        fc_normalized = (cutoff_freq + (transition_band / 2)) / sampling_freq
        wc = 2 * np.pi * fc_normalized 
        hd[n == 0] = 2 * fc_normalized 
        hd[n != 0] = (2 * fc_normalized) * (np.sin(n[n != 0] * wc) / (n[n != 0] * wc))

        if window_type == "Rectangular":
            w = np.ones_like(n)
        elif window_type == "Hanning":
            w = 0.5 + 0.5 * np.cos(2 * np.pi * n / N)
        elif window_type == "Hamming":
            w = 0.54 + 0.46 * np.cos(2 * np.pi * n / N)
        elif window_type == "Blackman":
            w = 0.42 + 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))

        h = hd * w
        coefficients = [(int(idx), coeff) for idx, coeff in zip(n, h)]
        convIndices, convSamples = self.convFilter(indices, samples, [idx for idx, _ in coefficients] ,[coeff for _, coeff in coefficients] )
        return convIndices, convSamples
    
    def resample(self):
        file_name = self.td_file_path_entry.get()
        test_file = self.td_test_file_entry.get()
        upsample_factor = int(self.upsampling_factor_entry.get())
        downsample_factor = int(self.downsampling_factor_entry.get())

        indices, samples = self.read_signal_file(file_name)

        if upsample_factor == 0 and downsample_factor == 0:
            print("Both upsampling and downsampling factors cannot be zero.")
            return None, None

        resampled_signal = None
        resampled_indices = None

        if upsample_factor == 0 and downsample_factor != 0:

            filtered_indeces, filtered_signal = self.Filtering(indices, samples)
            resampled_signal = filtered_signal[::downsample_factor]
            resampled_indices = np.arange(int(filtered_indeces[0]),int(filtered_indeces[0]+len(resampled_signal)),1)
        elif upsample_factor != 0 and downsample_factor == 0:
            upsampled_signal = np.zeros(len(samples) * upsample_factor)
            upsampled_signal[::upsample_factor] = samples
            upsampled_indices = list(np.arange(indices[0], len(upsampled_signal), 1))
            filtered_indices, resampled_signal = self.Filtering(upsampled_indices, upsampled_signal)

            while len(resampled_signal) > 0 and resampled_signal[-1] == 0:
                resampled_signal = resampled_signal[:-1]
                filtered_indices = filtered_indices[:-1]

            resampled_indices = list(np.arange(int(filtered_indices[0]), int(filtered_indices[0] + len(resampled_signal)), 1))

        else:
            upsampled_signal = np.zeros(len(samples) * upsample_factor)
            upsampled_signal[::upsample_factor] = samples
            upsampled_indices = list(np.arange(indices[0], len(upsampled_signal), 1))
            filtered_indices, filtered_signal = self.Filtering(upsampled_indices, upsampled_signal)

            while len(filtered_signal) > 0 and filtered_signal[-1] == 0:
                filtered_signal = filtered_signal[:-1]
                filtered_indices = filtered_indices[:-1]

            resampled_signal = filtered_signal[::downsample_factor]
            resampled_indices = np.arange(int(filtered_indices[0]), int(filtered_indices[0] + len(resampled_signal)), 1)

        resampled_indices = np.ravel(resampled_indices)
        resampled_signal = np.ravel(resampled_signal)
        for i in range(len(resampled_signal)):
            print(f"{resampled_indices[i]} {resampled_signal[i]:.9f}")
        print(len(resampled_indices))

        self.Compare_Signals(test_file, resampled_indices, resampled_signal)

        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.plot(indices, samples, label="Original Signal")
        plt.title("Original Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(resampled_indices, resampled_signal, label="Resampled Signal", color="orange")
        plt.title("Resampled Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.tight_layout()
        plt.show()

        return resampled_indices, resampled_signal

    def Compare_Signals(self, file_name,Your_indices,Your_samples):      
        expected_indices=[]
        expected_samples=[]
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                # process line
                L=line.strip()
                if len(L.split(' '))==2:
                    L=line.split(' ')
                    V1=int(L[0])
                    V2=float(L[1])
                    expected_indices.append(V1)
                    expected_samples.append(V2)
                    line = f.readline()
                elif len(L.split('  '))==2:
                    L=line.split('  ')
                    V1=int(L[0])
                    V2=float(L[1])
                    expected_indices.append(V1)
                    expected_samples.append(V2)
                    line = f.readline()
                else:
                    break
        print("Current Output Test file is: ")
        print(file_name)

        if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
            print("Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(Your_indices)):
            if(Your_indices[i]!=expected_indices[i]):
                print("Test case failed, your signal have different indicies from the expected one") 
                return
        for i in range(len(expected_samples)):
            if abs(Your_samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                print("Test case failed, your signal have different values from the expected one") 
                return
        print("Test case passed successfully")
    
    def convFilter(self, signal_indices, signal_samples, filter_indices, filter_coefficients):

        start_index = int(signal_indices[0] + filter_indices[0])
        end_index = int(signal_indices[-1] + filter_indices[-1])
        convolved_indices = list(range(start_index, end_index + 1))

        convolved_samples = []

        for i in convolved_indices:
            conv_sum = 0.0
            for k in range(len(filter_coefficients)):
                signal_index = i - filter_indices[k]
                if signal_index in signal_indices:
                    signal_idx = signal_indices.index(signal_index)
                    conv_sum += signal_samples[signal_idx] * filter_coefficients[k]
            convolved_samples.append(conv_sum)

        return convolved_indices, convolved_samples

    def ConvTest(self, Your_indices,Your_samples): 
        """
        Test inputs
        InputIndicesSignal1 =[-2, -1, 0, 1]
        InputSamplesSignal1 = [1, 2, 1, 1 ]
        
        InputIndicesSignal2=[0, 1, 2, 3, 4, 5 ]
        InputSamplesSignal2 = [ 1, -1, 0, 0, 1, 1 ]
        """
        
        expected_indices=[-2, -1, 0, 1, 2, 3, 4, 5, 6]
        expected_samples = [1, 1, -1, 0, 0, 3, 3, 2, 1 ]

        
        if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
            print("Conv Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(Your_indices)):
            if(Your_indices[i]!=expected_indices[i]):
                print("Conv Test case failed, your signal have different indicies from the expected one") 
                return
        for i in range(len(expected_samples)):
            if abs(Your_samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                print("Conv Test case failed, your signal have different values from the expected one") 
                return
        print("Conv Test case passed successfully")

    def smoothing(self):
        file_path = self.six_file_path_entry.get()
        test_file = self.six_test_file_entry.get()
        smoothedSignal=[]
        index, samples = self.read_signal_file(file_path)
        windowSize = int(self.windowSizeEntry.get())
        for i in range(0, len(samples)-windowSize + 1):
            Smoothedvalue = 0
            for j in range(i, i+windowSize):
                Smoothedvalue += samples[j]
            smoothedSignal.append(Smoothedvalue/windowSize)

        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.plot(samples)
        plt.title("Original Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(smoothedSignal)
        plt.title("Smoothed Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

        self.SignalSamplesAreEqual(test_file,index,smoothedSignal)
        return smoothedSignal
    
    def convolution(self):
        file_path1 = self.six_file_path_entry.get()
        file_path2 = self.six_file_path_entry2.get()
        index1, samples1 = self.read_signal_file(file_path1)
        index2, samples2 = self.read_signal_file(file_path2)

        max_index = index1[-1] + index2[-1]
        min_index = index1[0] + index2[0]
        convSignal = []
        outInd = []
        for n in range(int(min_index), int(max_index)+1):
            value = 0
            for k in range(int(min_index), int(max_index)+1):  # max+1
                if ((n-k) < 0):
                    continue

                if (k > index1[len(index1) - 1]):
                    xOfn = 0
                else:
                    xOfn = samples1[index1.index(k)]

                if ((n - k) > index2[len(index2) - 1]):
                    hOfn = 0
                else:
                    hOfn = samples2[index2.index(n-k)]

                value += xOfn * hOfn

            outInd.append(n)
            convSignal.append(value)
    
        plt.figure(figsize=(8, 6))
        plt.subplot(3, 1, 1)
        plt.plot(index1, samples1)
        plt.title("Original Signal 1")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.subplot(3, 1, 2)
        plt.plot(index2, samples2)
        plt.title("Original Signal 2")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        
        plt.subplot(3, 1, 3)
        plt.plot(outInd, convSignal)
        plt.title("Convolveed Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

        self.ConvTest(outInd,convSignal)
        return convSignal

    def correlation(self):
        file_path1 = self.six_file_path_entry.get()
        file_path2 = self.six_file_path_entry2.get()
        test_file = self.six_test_file_entry.get()
        index1, samples1 = self.read_signal_file(file_path1)
        index2, samples2 = self.read_signal_file(file_path2)

        r = []
        correlationsignal = []

        N = len(samples1)
        r.clear()

        for j in range(0, N):
            value = 0
            for n in range(0, N):
                d = n + j
                if (d >= N):
                    d -= N
                value += (samples1[n] * samples2[d])
            r.append(value / N)

        denominator = 0
        y1square = 0
        y2square = 0

        for i in range(0, N):
            y1square += np.power(samples1[i], 2)
            y2square += np.power(samples2[i], 2)

        denominator = (np.sqrt(y1square * y2square)) / N

        correlationsignal.clear()
        for i in r:
            correlationsignal.append(i / denominator)

        # plt.figure(figsize=(8, 6))
        # plt.subplot(3, 1, 1)
        # plt.plot(index1, samples1)
        # plt.title("Original Signal 1")
        # plt.xlabel("Index")
        # plt.ylabel("Amplitude")

        # plt.subplot(3, 1, 2)
        # plt.plot(index2, samples2)
        # plt.title("Original Signal 2")
        # plt.xlabel("Index")
        # plt.ylabel("Amplitude")
        
        # plt.subplot(3, 1, 3)
        # plt.plot(index1, correlationsignal)
        # plt.title("Convoluted Signal")
        # plt.xlabel("Index")
        # plt.ylabel("Amplitude")

        # plt.tight_layout()
        # plt.show()

        # self.Compare_Signals(test_file,index2,correlationsignal)
        print ("lensamples1")
        print (len(samples1))
        print ("lencorrsig")
        print (len(correlationsignal))
        return correlationsignal

    def remove_dc_component_frequency(self):
        file_path = self.six_file_path_entry.get()
        test_file = self.six_test_file_entry.get()
        index, samples = self.read_signal_file(file_path)
        amp, phase = self.run_frequency_domain_dc(0 , samples)
        complex_numbers = []
        for i in range(len(amp)):
            real_part = amp[i] * np.cos(float(phase[i]))
            imaginary_part = amp[i] * np.sin(float(phase[i]))
            complex_number = real_part + 1j * imaginary_part
            complex_numbers.append(complex_number)

        complex_numbers[0] = complex(0, 0)
        signal_without_dc = self.run_frequency_domain_dc(1 , complex_numbers)
        
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.plot(index, samples)
        plt.title("Original Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(index, signal_without_dc)
        plt.title("Signal Without DC")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

        self.SignalSamplesAreEqual(test_file,index,signal_without_dc)

        return signal_without_dc

    def remove_dc_component_time(self):
        file_path = self.six_file_path_entry.get()
        test_file = self.six_test_file_entry.get()
        index, samples = self.read_signal_file(file_path)

        total_sum = sum(samples)
        num_samples = len(samples)
        dc_component = total_sum / num_samples

        signal_without_dc = [sample - dc_component for sample in samples]

        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.plot(index, samples)
        plt.title("Original Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(index, signal_without_dc)
        plt.title("Signal Without DC")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

        self.SignalSamplesAreEqual(test_file,index,signal_without_dc)

        return signal_without_dc
    
    def fold_signal(self):
        file_path = self.td_file_path_entry.get()
        test_file = self.td_test_file_entry.get()
        

        index, samples = self.read_signal_file(file_path)
        folded_amplitudes = list(reversed(samples))

        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.plot(index, samples)
        plt.title("Original Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(index, folded_amplitudes)
        plt.title("Folded Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()
        self.SignalSamplesAreEqual(test_file,index,folded_amplitudes)
        return index, folded_amplitudes

    def shift_folded(self):
        file_path = self.td_file_path_entry.get()
        test_file = self.td_test_file_entry.get()

        index, samples = self.read_signal_file(file_path)
        folded_amplitudes = list(reversed(samples))  

        k_steps=float(self.steps_entry.get())
        
        for i in range(len(samples)): #iterate on indicies
            index[i] = index[i] + k_steps
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.plot(index, samples)
        plt.title("Original Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(index, folded_amplitudes)
        plt.title("Folded Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()

        self.Shift_Fold_Signal(test_file,index,folded_amplitudes)

    def shift_only(self):
        file_path = self.td_file_path_entry.get()
        test_file = self.td_test_file_entry.get()
        indexShifted=[]
        index, samples = self.read_signal_file(file_path)
        k_steps=float(self.steps_entry.get())
        
        for i in range(len(samples)): #iterate on indicies
            indexShifted[i] = index[i] + k_steps
        self.Shift_Fold_Signal(test_file,index,samples)
        plt.figure(figsize=(8, 6))
        plt.subplot(2, 1, 1)
        plt.plot(index, samples)
        plt.title("Original Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(indexShifted, samples)
        plt.title("Shifted Signal")
        plt.xlabel("Index")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()
        self.Shift_Fold_Signal(test_file,index,samples)

    def Shift_Fold_Signal(self,file_name,Your_indices,Your_samples):      
        expected_indices=[]
        expected_samples=[]
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                # process line
                L=line.strip()
                if len(L.split(' '))==2:
                    L=line.split(' ')
                    V1=int(L[0])
                    V2=float(L[1])
                    expected_indices.append(V1)
                    expected_samples.append(V2)
                    line = f.readline()
                else:
                    break
        print("Current Output Test file is: ")
        print(file_name)
        print("\n")
        if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
            print("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(Your_indices)):
            if(Your_indices[i]!=expected_indices[i]):
                print("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one") 
                return
        for i in range(len(expected_samples)):
            if abs(Your_samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                print("Shift_Fold_Signal Test case failed, your signal have different values from the expected one") 
                return
        print("Shift_Fold_Signal Test case passed successfully")
        
    def DerivativeSignal(self):
        InputSignal=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
        expectedOutput_first = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        expectedOutput_second = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        FirstDrev = [float(InputSignal[i]) -float(InputSignal[i-1] ) for i in range(1, len(InputSignal))] 
        # Y2(n)= x(n+1)-2x(n)+x(n-1)
        SecondDrev = [float(InputSignal[i+1]) - 2*float(InputSignal[i]) + float(InputSignal[i-1]) for i in range(1, len(InputSignal)-1)]
        # Display First Derivative
        print("First Derivative:")
        print(f"Y1(n) = {FirstDrev}")
        print("\nSecond Derivative:")
        print(f"Y2(n) = {SecondDrev}") 
        if( (len(FirstDrev)!=len(expectedOutput_first)) or (len(SecondDrev)!=len(expectedOutput_second))):
            print("mismatch in length") 
            return
        first=second=True
        for i in range(len(expectedOutput_first)):
            if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
                continue
            else:
                first=False
                print("1st derivative wrong")
                return
        for i in range(len(expectedOutput_second)):
            if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
                continue
            else:
                second=False
                print("2nd derivative wrong") 
                return
        if(first and second):
            print("Derivative Test case passed successfully")
        else:
            print("Derivative Test case failed")
        return

  

    def remove_file(self):
        selected_indices = self.file_listbox.curselection()
        for index in reversed(selected_indices):
            self.file_listbox.delete(index)

    def SignalSamplesAreEqual(self,file_name, indices, samples):
        expected_indices = []
        expected_samples = []
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()

            while line:
                # process line
                L = line.strip()
                if len(L.split(' ')) == 2:
                    L = line.split(' ')
                    V1 = int(L[0])
                    V2 = float(L[1])
                    expected_indices.append(V1)
                    expected_samples.append(V2)
                    line = f.readline()
                else:
                    break

        if len(expected_samples) != len(samples):
            print("Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(expected_samples)):
            if abs(samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                print("Test case failed, your signal have different values from the expected one")
                return
        print("Test case passed successfully")

    global summ
    
    def summ(self, array):
        if not array:
            return []
        
        result = [array[0]]
        for i in range(1, len(array)):
            result.append(result[-1] + array[i])
    
        return result
    
    def execute_operation(self):
        selected_files = self.file_listbox.get(0, tk.END)
        if not selected_files:
            messagebox.showerror("Error", "Please select at least one file.")
            return

        operation = self.operation_var.get()
        constant = self.constant_var.get()

        signals = []
        for file in selected_files:
            signal_type_str, is_periodic, data = self.read_signal_file(file)
            if signal_type_str == 'Time':
                t, signal = zip(*data)
            else:
                t, signal, _ = zip(*data)
            signals.append((np.array(t), np.array(signal)))

        if not signals:
            messagebox.showerror("Error", "No valid signals loaded.")
            return

        result_t, result_signal = self.perform_operation(signals, operation, constant)
        self.save_output_file("Saved.txt", result_t, result_signal)
        self.plot_signal(result_t, result_signal, f"Result of {operation}")
        #self.SignalSamplesAreEqual("C:/Users/salah/OneDrive/Desktop/DSP/Output files/output accumulation for signal1.txt", result_t, result_signal)
        self.current_signal = (result_t, result_signal)

    def save_output_file(self,file_name, t, signal):
        with open(file_name, 'w') as f:
            f.write("Time (s) Amplitude\n")
            for time, amplitude in zip(t, signal):
                f.write(f"{time:.6f} {amplitude:.6f}\n")

    def perform_operation(self, signals, operation, constant):
        if operation == "Addition":
            result = np.zeros_like(signals[0][1])
            for _, signal in signals:
                result += signal
            return signals[0][0], result
        elif operation == "Subtraction":
            result = signals[0][1].copy()
            for _, signal in signals[1:]:
                result -= signal
                result = abs(result)
            return signals[0][0], result
        elif operation == "Multiplication":
            return signals[0][0], signals[0][1] * constant
        elif operation == "Squaring":
            return signals[0][0], np.square(signals[0][1])
        elif operation == "Normalization":
            signal = signals[0][1]
            result = 2 * (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) - 1
            return signals[0][0], result
        elif operation == "Accumulation":
            return signals[0][0], summ(signals[0][1])
        else:
            raise ValueError(f"Unsupported operation: {operation}")

    def read_signal_file(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

            signal_type = int(lines[0].strip())
            signal_type_str = 'Time' if signal_type == 0 else 'Frequency'

            is_periodic = bool(int(lines[1].strip()))

            N1 = int(lines[2].strip())

            data = []
            if signal_type == 0:
                for i in range(3, 3 + N1):
                    index, amplitude = map(float, lines[i].strip().split())
                    data.append((index, amplitude))
            else:
                for i in range(3, 3 + N1):
                    frequency, amplitude, phase_shift = map(float, lines[i].strip().split())
                    data.append((frequency, amplitude, phase_shift))

        return signal_type_str, is_periodic, data

    def plot_signal(self, t, signal, title):
        plt.figure(figsize=(10, 4))
        plt.plot(t, signal)
        plt.title(title)
        plt.xlabel('Time (s)' if len(t) == len(signal) else 'Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.show()

    def create_quantization_page(self):
        quantization_frame = ttk.Frame(self.notebook)
        self.notebook.add(quantization_frame, text="Quantization")

        file_frame = ttk.LabelFrame(quantization_frame, text="File Selection", padding=10)
        file_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(file_frame, text="Input File:").grid(row=0, column=0, sticky="w", pady=5)
        self.fd_file_path_entry = ttk.Entry(file_frame, width=50)
        self.fd_file_path_entry.grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.select_file(self.fd_file_path_entry)).grid(row=0, column=2)

        ttk.Label(file_frame, text="Test File:").grid(row=1, column=0, sticky="w", pady=5)
        self.fd_test_file_entry = ttk.Entry(file_frame, width=50)
        self.fd_test_file_entry.grid(row=1, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=lambda: self.select_file(self.fd_test_file_entry)).grid(row=1, column=2)

        param_frame = ttk.LabelFrame(quantization_frame, text="Quantization Parameters", padding=10)
        param_frame.pack(fill="x", padx=10, pady=5)


        self.is_bits_var = tk.BooleanVar(value=True)
        ttk.Label(param_frame, text="Input Type:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Radiobutton(param_frame, text="Bits", variable=self.is_bits_var, value=True).grid(row=0, column=1)
        ttk.Radiobutton(param_frame, text="Levels", variable=self.is_bits_var, value=False).grid(row=0, column=2)


        ttk.Label(param_frame, text="Levels/Bits:").grid(row=1, column=0, sticky="w", pady=5)
        self.levels_or_bits_entry = ttk.Entry(param_frame, width=20)
        self.levels_or_bits_entry.grid(row=1, column=1, columnspan=2, sticky="w")

       
        ttk.Button(quantization_frame, text="Run Quantization", command=self.run_quantization).pack(pady=20)

    def select_file(self, entry):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            entry.delete(0, tk.END)
            entry.insert(0, file_path)

    def quantization(self, array, levels_or_bits, is_bits):
        if not array:
            raise ValueError("Input array is empty.")

        min_value = min(array)
        max_value = max(array)

        if is_bits:
            bits = levels_or_bits
            levels = 2 ** bits
        else:
            levels = levels_or_bits
            bits = int(math.log2(levels))

        delta = (max_value - min_value) / levels 

        quantized_values = []
        encoded_values = []
        interval_indices = []
        quant_error = []

        
        interval_boundaries = [min_value + i * delta for i in range(levels + 1)]
        midpoints = [(interval_boundaries[i] + interval_boundaries[i + 1]) / 2 for i in range(levels)]

        for sample in array:
            
            for i in range(levels):
                if interval_boundaries[i] <= sample < interval_boundaries[i + 1]:
                    quantized_value = midpoints[i]  
                    display_index = i + 1 if not is_bits else i
                    error = quantized_value - sample

                    quantized_values.append(round(quantized_value, 3))
                    encoded_values.append(format(i, f'0{bits}b'))
                    interval_indices.append(display_index)
                    quant_error.append(round(error, 3))
                    break
            else:
                
                quantized_value = midpoints[-1]
                display_index = levels if not is_bits else levels - 1
                error = quantized_value - sample

                quantized_values.append(round(quantized_value, 3))
                encoded_values.append(format(levels - 1, f'0{bits}b'))
                interval_indices.append(display_index)
                quant_error.append(round(error, 3))

        return quantized_values, encoded_values, interval_indices, quant_error
    
    def transform_testing(self,file_path,value1,value2):
       
        testsignaltype,test=self.read_signal_file_fourier(file_path)
        expvalue1 = [sample[0] for sample in test]
        expvalue2 = [sample[1] for sample in test]
        if len(test) != len(value1):
            print("Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(test)):
            if abs(value1[i] - expvalue1[i]) < 0.01 & abs(value2[i] - expvalue2[i]) < 0.01:
                continue
            else:
                print("Test case failed, your signal have different values from the expected one")
                return
        print("Test case passed successfully")
        return

    def DCT(self):
        file_path = self.fd_file_path_entry.get()
        test_file = self.fd_test_file_entry.get()
        if not file_path or not test_file:
            messagebox.showerror("Error", "Please select both input and test files.")
            return
        
        t, signal = self.read_signal_file(file_path)
        result=[]
        N=len(signal)
        for k in range(0, N):
            summation = 0
            for n in range(0, N):
                summation += signal[n] * math.cos((math.pi / (4 * N) * (2 * n - 1) * (2 * k - 1)))
            result.append(math.sqrt(2 / N) * summation)
        print(result)
        self.SignalSamplesAreEqual(test_file, t, result)
        num_lines = int(self.coeff_entry.get())
        if num_lines is not None:
            num_lines = min(num_lines, len(t))
            tsave = t[:num_lines]
            ssave = signal[:num_lines]
        else:
            messagebox.showerror("Error", "Please input the number of coefficients.")
            return
        with open("Saved DCT.txt", 'w') as f:
            f.write("Time (s) Amplitude\n")
            for time, amplitude in zip(tsave, ssave):
                f.write(f"{time:.6f} {amplitude:.6f}\n")

        return result

    def read_signal_file(self, filename):
        with open(filename, 'r') as file:
            t = []
            signal = []
            lines = file.readlines()
            for line in lines[3:]:  
                if len(line.strip().split()) >= 2:
                    values = line.strip().split()
                    t.append(float(values[0]))
                    signal.append(float(values[1]))
        return t, signal
    
    def run_frequency_domain(self):
        try:
            file_path = self.fd_file_path_entry.get()
            test_file = self.fd_test_file_entry.get()
            if not file_path or not test_file:
                messagebox.showerror("Error", "Please select both input and test files.")
                return

            fs = int(self.fs_entry.get())
            signaltype, x = self.read_signal_file_fourier(file_path)
            
            N = len(x)
            A = []
            P = []
            frequencies = [(2 * np.pi * fs * k) / N for k in range(N)]

            if signaltype == 0:
                # Extract amplitude values for Fourier transform
                amps = [sample[1] for sample in x]
                X = []

                for k in range(N):
                    real_part = 0
                    imag_part = 0

                    for n in range(N):
                        angle = -2 * np.pi * k * n / N
                        real_part += amps[n] * np.cos(angle)
                        imag_part += amps[n] * np.sin(angle)
                    
                    X.append(complex(real_part, imag_part))

                # Calculate amplitude and phase for each frequency component
                for index in X:
                    real = index.real
                    imag = index.imag
                    A.append(np.sqrt(real**2 + imag**2))
                    P.append(np.arctan2(imag, real))

                # Plot amplitude and phase spectra
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.stem(frequencies, A)
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Amplitude")
                plt.title("Amplitude Spectrum")
                
                plt.subplot(2, 1, 2)
                plt.stem(frequencies, P)
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Phase (Radians)")
                plt.title("Phase Spectrum")
                
                plt.tight_layout()
                plt.show()

                #self.SignalSamplesAreEqual(test_file,phaseshifts,amps)

                #for p in range (printlen):
                #    print(phaseshifts[p],"\n",amps[p])
                print (P)
                print (A)
                # self.transform_testing(test_file,phaseshifts,amps)

            else:
                # Inverse Fourier transform (time-domain reconstruction)
                inverse_init_values = []
                for sample in x:
                    amp = sample[0]
                    phase = sample[1]
                    real_part = amp * np.cos(phase)
                    imag_part = amp * np.sin(phase)
                    inverse_init_values.append(complex(real_part, imag_part))

                LEN = len(inverse_init_values)
                TUP = []
                
                for n in range(LEN):
                    sum_value = 0
                    for k in range(LEN):
                        sum_value += inverse_init_values[k] * np.exp(2j * np.pi * k * n / LEN)

                    TUP.append(sum_value / LEN)

                original_signal = np.real(TUP)

                # Plot reconstructed time-domain signal
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.plot(original_signal, label="Analog Signal")
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.title("Analog Signal")
                
                plt.subplot(2, 1, 2)
                plt.stem(range(len(original_signal)), original_signal)
                plt.xlabel("Sample Number")
                plt.ylabel("Amplitude")
                plt.title("Digital Signal")
                plt.tight_layout()
                plt.show()
                
                #value1=[n for n in range(len(original_signal))]
                #value2=[sample[0] for sample in original_signal]
                #printlen= len(value1)
                #with open("output.txt", "w") as output_file:

                #    for e, q in zip(value1, value2):
                #        output_file.write(f"{e} {q:.3f}\n")

                #self.SignalSamplesAreEqual(test_file,value1,value2)
                #self.transform_testing(test_file,value1,value2)
                print(original_signal)

        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def run_frequency_domain_dc(self,type,signal):
        N = len(signal)
        A = []
        P = []
        if type == 0:
            X = []

            for k in range(N):
                real_part = 0
                imag_part = 0

                for n in range(N):
                    angle = -2 * np.pi * k * n / N
                    real_part += signal[n] * np.cos(angle)
                    imag_part += signal[n] * np.sin(angle)
                
                X.append(complex(real_part, imag_part))

            for index in X:
                real = index.real
                imag = index.imag
                A.append(np.sqrt(real**2 + imag**2))
                P.append(np.arctan2(imag, real))
            return A, P
        else:

            LEN = len(signal)
            TUP = []
            
            for n in range(LEN):
                sum_value = 0
                for k in range(LEN):
                    sum_value += signal[k] * np.exp(2j * np.pi * k * n / LEN)

                TUP.append(sum_value / LEN)

            original_signal = np.real(TUP)
            return original_signal


    def read_signal_file_fourier(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

            is_periodic = bool(int(lines[0].strip()))
            signal_type = int(lines[1].strip())
            N1 = int(lines[2].strip())

            data = []
            if signal_type == 0:
                for i in range(3, 3 + N1):
                    index, amplitude = map(float, lines[i].strip().split())
                    data.append((index, amplitude))
            else:
                for i in range(3, 3 + N1):
                    line = lines[i].strip()
                    if ',' in line:
                        parts = [p.replace('f', '') for p in line.split(',')]
                    else:
                        parts = line.strip().split()
                        parts = [p.replace('f', '') for p in parts]
                    amplitude, phase_shift = map(float, parts)
                    data.append((amplitude, phase_shift))

        return signal_type, data 

    def run_quantization(self):
        try:
            file_path = self.quant_file_path_entry.get()
            test_file = self.quant_test_file_entry.get()

            if not file_path or not test_file:
                messagebox.showerror("Error", "Please select both input and test files.")
                return

            levels_or_bits = int(self.levels_or_bits_entry.get())
            is_bits = self.is_bits_var.get()

            t, signal = self.read_signal_file(file_path)

            quantized_values, encoded_values, interval_indices, quant_error = self.quantization(
                signal, levels_or_bits, is_bits
            )


            with open("output.txt", "w") as output_file:
                if is_bits:
                    for e, q in zip(encoded_values, quantized_values):
                        output_file.write(f"{e} {q:.3f}\n")
                    self.QuantizationTest1(test_file, encoded_values, quantized_values)
                else:
                    for idx, e, q, err in zip(interval_indices, encoded_values, quantized_values, quant_error):
                        output_file.write(f"{idx} {e} {q:.3f} {err:.3f}\n")
                    self.QuantizationTest2(test_file, interval_indices, encoded_values, quantized_values, quant_error)

            self.plot_quantization_results(t, signal, quantized_values)
            messagebox.showinfo("Success", "Quantization completed successfully. Results saved to output.txt")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_quantization_results(self, t, original_signal, quantized_signal):
        plt.figure(figsize=(12, 6))
        plt.plot(t, original_signal, 'b-', label='Original Signal', alpha=0.7)
        plt.plot(t, quantized_signal, 'r-', label='Quantized Signal', alpha=0.7)
        plt.title('Signal Quantization Results')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def QuantizationTest1(self, file_name, Your_EncodedValues, Your_QuantizedValues):
        expectedEncodedValues = []
        expectedQuantizedValues = []
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                # process line
                L = line.strip()
                if len(L.split(' ')) == 2:
                    L = line.split(' ')
                    V2 = str(L[0])
                    V3 = float(L[1])
                    expectedEncodedValues.append(V2)
                    expectedQuantizedValues.append(V3)
                    line = f.readline()
                else:
                    break
        if ((len(Your_EncodedValues) != len(expectedEncodedValues)) or (
                len(Your_QuantizedValues) != len(expectedQuantizedValues))):
            print("QuantizationTest1 Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(Your_EncodedValues)):
            if (Your_EncodedValues[i] != expectedEncodedValues[i]):
                print(
                    "QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one")
                return
        for i in range(len(expectedQuantizedValues)):
            if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
                continue
            else:
                print(
                    "QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one")
                return
        print("QuantizationTest1 Test case passed successfully")

    def QuantizationTest2(self, test_file, Your_IntervalIndices, Your_EncodedValues, Your_QuantizedValues,
                          Your_SampledError):
        expectedIntervalIndices = []
        expectedEncodedValues = []
        expectedQuantizedValues = []
        expectedSampledError = []
        with open(test_file, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                L = line.strip()
                if len(L.split(' ')) == 4:
                    L = line.split(' ')
                    V1 = int(L[0])
                    V2 = str(L[1])
                    V3 = float(L[2])
                    V4 = float(L[3])
                    expectedIntervalIndices.append(V1)
                    expectedEncodedValues.append(V2)
                    expectedQuantizedValues.append(V3)
                    expectedSampledError.append(V4)
                    line = f.readline()
                else:
                    break
        if (len(Your_IntervalIndices) != len(expectedIntervalIndices) or
                len(Your_EncodedValues) != len(expectedEncodedValues) or
                len(Your_QuantizedValues) != len(expectedQuantizedValues) or
                len(Your_SampledError) != len(expectedSampledError)):
            print("QuantizationTest2 Test case failed, your signal has a different length from the expected one")
            return
        for i in range(len(Your_IntervalIndices)):
            if Your_IntervalIndices[i] != expectedIntervalIndices[i]:
                print("QuantizationTest2 Test case failed, your indices are different from the expected ones")
                return
        for i in range(len(Your_EncodedValues)):
            if Your_EncodedValues[i] != expectedEncodedValues[i]:
                print("QuantizationTest2 Test case failed, your EncodedValues are different from the expected ones")
                return
        for i in range(len(expectedQuantizedValues)):
            if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
                continue
            else:
                print("QuantizationTest2 Test case failed, your QuantizedValues are different from the expected ones")
                return
        for i in range(len(expectedSampledError)):
            if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
                continue
            else:
                print(
                    "QuantizationTest2 Test case failed, your SampledError values are different from the expected ones")
                return
        print("QuantizationTest2 Test case passed successfully")

if __name__ == "__main__":
    root = tk.Tk()
    aplt = SignalGenerator(root)
    root.mainloop()
