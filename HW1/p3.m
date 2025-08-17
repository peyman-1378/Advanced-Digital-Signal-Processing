% Parameters
clear all, close all, clc;
Fs = 22000;            
N = 22000;             % Number of samples
gamma = (pi/16 + 3*pi/(8*N))/2;           
phi = 0;               
a = 1;                 
SNR_dB = 10; % Signal-to-noise ratio in dB
SNR = 10^(SNR_dB/10); % Linear SNR
sigma_w = a / sqrt(SNR);         % noise standard deviation

% Generate time vector and noisy signal
n = 0:N-1;
w = sigma_w * randn(1, N);   
x = a * cos(gamma * n.^2 + phi) + w;

% Play the sound 
sound(x, Fs);

% STFT parameters
window = hamming(256);   
noverlap = 128;          
nfft = 512;             

% Compute STFT
[S, F, T] = spectrogram(x, window, noverlap, nfft, Fs);

% Instantaneous frequency estimation
[~, maxIndex] = max(abs(S), [], 1); 
instantaneous_freq = F(maxIndex);

% Plot the estimated instantaneous frequency
figure;
plot(T, instantaneous_freq);
xlabel('Time (s)');
ylabel('Instantaneous Frequency (Hz)');
title('Estimated Instantaneous Frequency of FM Signal');