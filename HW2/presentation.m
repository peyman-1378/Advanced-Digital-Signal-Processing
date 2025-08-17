clc;clear; close all;

duration = 10;
r0 = 3;
d = 0.18;
c = 343;
omega0 = pi;
mic1_position = [-d/2, 0];
mic2_position = [d/2, 0];

[audio, fs] = audioread('A.mp3');
audio = audio(1200000:end,1);

t = (0:1/fs:duration-1/fs)';

x_s = r0 * cos(omega0 * t);
y_s = r0 * sin(omega0 * t);

d1 = sqrt((x_s - mic1_position(1)).^2 + (y_s - mic1_position(2)).^2);
d2 = sqrt((x_s - mic2_position(1)).^2 + (y_s - mic2_position(2)).^2);

source_signal = interp1((1:length(audio))/fs, audio, t, 'pchip');

delay1 = d1 / c;
delay2 = d2 / c;

minLength = min(length(source_signal), min(length(d1), length(d2)));
source_signal = source_signal(1:minLength);
delay1 = delay1(1:minLength);
delay2 = delay2(1:minLength);

s1 = zeros(size(source_signal));
s2 = zeros(size(source_signal));

for i = 1:minLength
    s1(i) = source_signal(max(1, i - round(delay1(i)*fs)));
    s2(i) = source_signal(max(1, i - round(delay2(i)*fs)));
end

att_factor1 = (r0 ./ d1).^2;
att_factor2 = (r0 ./ d2).^2;

s1 = s1 .* att_factor1;
s2 = s2 .* att_factor2;

r1 = s1 + 0.01 * randn(size(s1));
r2 = s2 + 0.01 * randn(size(s2));

figure
plot(t,r1)
hold on
plot(t,r2)

stereo_signal = [r1, r2];
sound(stereo_signal, fs);

%% a)

deltaTau_c = delay2 - delay1;

figure;
plot(t, deltaTau_c);
xlabel('Time (s)');
ylabel('Difference in Time of Arrival (s)');
title('Difference Time of Arrival (DToA) vs Time');

noise_levels = logspace(-4, -1, 100); 
mse_values_s1 = zeros(size(noise_levels));
mse_values_s2 = zeros(size(noise_levels));
snr_values_s1 = zeros(size(noise_levels));
snr_values_s2 = zeros(size(noise_levels));

for i = 1:length(noise_levels)
    
    noise_s1 = noise_levels(i) * randn(size(s1));
    noise_s2 = noise_levels(i) * randn(size(s2));
    
    noisy_s1 = s1 + noise_s1;
    noisy_s2 = s2 + noise_s2;
    
    signal_power_s1 = mean(s1.^2);
    signal_power_s2 = mean(s2.^2);
    
    noise_power_s1 = mean(noise_s1.^2);
    noise_power_s2 = mean(noise_s2.^2);
    
    snr_values_s1(i) = 10 * log10(signal_power_s1 / noise_power_s1);
    snr_values_s2(i) = 10 * log10(signal_power_s2 / noise_power_s2);
    
    mse_values_s1(i) = mean((s1 - noisy_s1).^2);
    mse_values_s2(i) = mean((s2 - noisy_s2).^2);
end

figure;
subplot(2,1,1); 
plot(snr_values_s1, pow2db(mse_values_s1), 'b');
xlabel('SNR for s1 (dB)');
ylabel('MSE for s1');
title('MSE vs. SNR for s1');
set(gca, 'XDir','reverse');
grid on;

subplot(2,1,2);
plot(snr_values_s2, mse_values_s2, 'r');
xlabel('SNR for s2 (dB)');
ylabel('MSE for s2');
title('MSE vs. SNR for s2');
set(gca, 'XDir','reverse');
grid on;

%% b)

inv_att_factor1 = 1 ./ att_factor1;
inv_att_factor2 = 1 ./ att_factor2;

r1_equalized = r1 .* inv_att_factor1;
r2_equalized = r2 .* inv_att_factor2;

mean_deltaTau_c = mean(deltaTau_c);
alpha1 = -mean_deltaTau_c / 2;
alpha2 = mean_deltaTau_c / 2;

minTime = max(t(1) - alpha1, t(1) - alpha2);
maxTime = min(t(end) - alpha1, t(end) - alpha2);

t_alpha1 = t(t >= minTime & t <= maxTime) + alpha1;
t_alpha2 = t(t >= minTime & t <= maxTime) + alpha2;

R1h = interp1(t, r1_equalized, t_alpha1, 'spline', 'extrap');
R2h = interp1(t, r2_equalized, t_alpha2, 'spline', 'extrap');

minLengthEqualized = min(length(R1h), length(R2h));
R1h = R1h(1:minLengthEqualized);
R2h = R2h(1:minLengthEqualized);

figure
plot(t_alpha1,R1h)
hold on
plot(t_alpha2,R2h)

stereo_signal_aligned_equalized = [R1h, R2h];
sound(stereo_signal_aligned_equalized, fs);

% audiowrite('Aligned_Equalized_Signal_1.wav', R1h_equalized, fs);
% audiowrite('Aligned_Equalized_Signal_2.wav', R2h_equalized, fs);

%% c)

Rh = 0.5 * R1h + 0.5 * R2h;

toa_values = zeros(size(noise_levels));
snr_values = zeros(size(noise_levels));

for i = 1:length(noise_levels)
    
    noise = noise_levels(i) * randn(size(Rh));
    Rh_noisy = Rh + noise;
    
    signal_power = mean(Rh.^2);
    noise_power = mean(noise.^2);
    
    snr_values(i) = 10 * log10(signal_power / noise_power);
    
    [cross_corr, lags] = xcorr(Rh_noisy, source_signal);
    
    [~, idx_max] = max(abs(cross_corr));
    toa_values(i) = lags(idx_max) / fs;
end

figure;
plot(snr_values, toa_values, 'o-');
xlabel('SNR (dB)');
ylabel('Time of Arrival (s)');
title('Estimated ToA vs SNR');
grid on;
