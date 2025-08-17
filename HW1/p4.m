%% 1.4
% Step 1: Load the Stereo Audio File
clc; clear; close all;
load('Hw2a.mat'); 

audio = Sin_d; 
fs = fs; 

% Step 2: Play the Audio
disp('Playing original audio...');
sound(audio, fs);
pause(length(audio)/fs + 2);

% Step 3: Visualize the Spectrogram
window_size = 1024;
overlap = 512;
nfft = 2048;
[S, F, T, P] = spectrogram(audio(:,1), window_size, overlap, nfft, fs, 'yaxis'); % Spectrogram of the first channel

% Plot the spectrogram
figure;
surf(T, F, 10*log10(P), 'edgecolor', 'none');
axis tight;
view(0, 90);
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram of Original Audio');
colorbar;

% Step 4: Estimate the Frequencies of the Sinusoids

% Using STFT to identify prominent frequencies
% Sum the power spectral density across time to find prominent frequencies
power_spectrum = sum(P, 2);

% Find peaks in the power spectrum
[~, peak_indices] = findpeaks(power_spectrum, 'MinPeakHeight', mean(power_spectrum));
prominent_freqs = F(peak_indices);

% Display identified frequencies
disp('Prominent frequencies (Hz):');
disp(prominent_freqs);

% Step 5: Remove the Sinusoids
% Design notch filters to remove the identified sinusoids

cleaned_audio = audio; 

for i = 1:length(prominent_freqs)
    d = designfilt('bandstopiir', 'FilterOrder', 2, ...
        'HalfPowerFrequency1', prominent_freqs(i) - 2, ...
        'HalfPowerFrequency2', prominent_freqs(i) + 2, ...
        'DesignMethod', 'butter', 'SampleRate', fs);
    cleaned_audio = filtfilt(d, cleaned_audio);
end

% Step 6: Save and Play the Cleaned Audio
disp('Playing cleaned audio...');
sound(cleaned_audio, fs);
pause(length(cleaned_audio)/fs + 2);


audiowrite('cleaned_audio.wav', cleaned_audio, fs);

% Visualize the spectrogram of the cleaned audio
[S_cleaned, F_cleaned, T_cleaned, P_cleaned] = spectrogram(cleaned_audio(:,1), window_size, overlap, nfft, fs, 'yaxis');

figure;
surf(T_cleaned, F_cleaned, 10*log10(P_cleaned), 'edgecolor', 'none');
axis tight;
view(0, 90);
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Spectrogram of Cleaned Audio');
colorbar;

