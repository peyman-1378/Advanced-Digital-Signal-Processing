%% 1.1 Noise generation

clc; clear all; close all;

% Parameters
rho_values = linspace(0, 0.99, 5);
N_values = linspace(20, 200, 10);
L = 500; % Number of samples for Monte Carlo
num_mc = 50; % Number of Monte Carlo iterations

MSE = zeros(length(N_values), length(rho_values));

for n_idx = 1:length(N_values)
    N = N_values(n_idx);
    for rho_idx = 1:length(rho_values)
        rho = rho_values(rho_idx);
        
        % True covariance matrix
        Cww = rho.^abs((1:N)' - (1:N));

        mse_accum = 0;
        for mc = 1:num_mc
            % Generate correlated noise
            L_chol = chol(Cww, 'lower');
            w = L_chol * randn(N, L);
            
            % Compute sample covariance
            C_hat = cov(w');
            
            % MSE calculation
            mse_accum = mse_accum + norm(C_hat - Cww, 'fro')^2 / numel(Cww);
        end
        
        % Average MSE over all Monte Carlo iterations
        MSE(n_idx, rho_idx) = mse_accum / num_mc;
    end
end

% Plotting MSE vs rho
figure;
hold on;
for n_idx = 1:length(N_values)
    plot(rho_values, pow2db(MSE(n_idx, :)), '-o', 'DisplayName', ['N = ' num2str(N_values(n_idx))]);
end
xlabel('\rho');
ylabel('MSE (dB)');
title('MSE vs \rho for Different Values of N');
legend('show', 'Location', 'northeast');
grid on;
hold off;

%% 1.2 Frequency estimation
% 1.2.1 L=1
% Uncorrelated noise

% Parameters
N_values = [100, 500, 1024];
omega_values = pi * [5./N_values; 10./N_values; 1/4 * ones(1,3); 1/2 * ones(1,3)];
SNR_dB = -20:5:20;
SNR = 10.^(SNR_dB/10);
num_realizations = 500;     % Number of Monte Carlo realizations

MSE_vs_SNR = zeros(length(SNR_dB), length(omega_values), length(N_values));
CRB_vs_SNR = zeros(size(MSE_vs_SNR));

% Loop over parameters
for i_N = 1:length(N_values)
    N = N_values(i_N);

    for i_omega = 1:length(omega_values)
        omega = omega_values(i_omega, i_N);

        for i_SNR = 1:length(SNR)
            snr = SNR(i_SNR);
            sigma_w = sqrt(1/snr);  % Noise standard deviation

            % Monte Carlo realizations
            omega_hat = zeros(num_realizations, 1);

            for i_realization = 1:num_realizations
                phi = 2*pi*rand();  % Random phase
                x = cos(omega*(0:N-1) + phi) + sigma_w * randn(1, N);

                % Frequency estimation
                X = abs(fft(x)).^2;
                [~, idx] = max(X);
                omega_hat(i_realization) = 2*pi*(idx-1)/N;
            end

            % Calculate MSE
            MSE_vs_SNR(i_SNR, i_omega, i_N) = mean((omega_hat - omega).^2);

            % Calculate CRB (single sinusoid with uncorrelated noise)
            CRB_vs_SNR(i_SNR, i_omega, i_N) = 24 * sigma_w^2 / (N*(N^2-1));
        end
    end
end

% Plotting
figure;
for i_N = 1:length(N_values)
    for i_omega = 1:length(omega_values)
        subplot(length(N_values), length(omega_values), (i_N-1)*length(omega_values) + i_omega);
        semilogy(SNR_dB, MSE_vs_SNR(:, i_omega, i_N), 'o-', SNR_dB, CRB_vs_SNR(:, i_omega, i_N), 'x-');
        title(['N = ' num2str(N_values(i_N)) ', \omega = ' num2str(omega_values(i_omega, i_N)/pi) '\pi']);
        xlabel('SNR (dB)');
        ylabel('MSE');
        legend('MSE', 'CRB');
        grid on;
    end
end
%% 1.2 Frequency estimation
% 1.2.1 L=1
% 2: correlated noise

% Parameters
N_values = [100, 500, 1024]; 
omega_values = pi * [5./N_values; 10./N_values; 1/4 * ones(1,3); 1/2 * ones(1,3)]; 
SNR_dB = -20:5:20;           
SNR = 10.^(SNR_dB/10);       
num_realizations = 500;     % Number of Monte Carlo realizations
rho = 0.9;                   

MSE_vs_SNR = zeros(length(SNR_dB), length(omega_values), length(N_values));
CRB_vs_SNR = zeros(size(MSE_vs_SNR));

for i_N = 1:length(N_values)
    N = N_values(i_N);

    for i_omega = 1:length(omega_values)
        omega = omega_values(i_omega, i_N);

        for i_SNR = 1:length(SNR)
            snr = SNR(i_SNR);
            
            % Calculate noise standard deviation (sigma_w) from SNR
            sigma_w = sqrt(1 / snr);  

            % Generate the covariance matrix Cww
            Cww = sigma_w^2 * rho.^abs((1:N)' - (1:N));
            
            % Cholesky decomposition of Cww
            L = chol(Cww, 'lower');

            % Monte Carlo realizations
            omega_hat = zeros(num_realizations, 1);
            for i_realization = 1:num_realizations
                phi = 2*pi*rand();  % Random phase
                 
                x = cos(omega*(0:N-1) + phi) + (L * randn(N, 1))';  % Generate signal + noise
                
                % Frequency estimation (ML)
                X = abs(fft(x)).^2;
                [~, idx] = max(X);
                omega_hat(i_realization) = 2*pi*(idx-1)/N;
            end

            % Calculate MSE
            MSE_vs_SNR(i_SNR, i_omega, i_N) = mean((omega_hat - omega).^2);
            
            % Calculate CRB (single sinusoid with correlated Gaussian noise)
            dx_domega = -(0:N-1).*sin(omega*(0:N-1) + phi); % Derivative of x w.r.t omega
            FIM = (dx_domega / Cww) * dx_domega'; % Fisher Information Matrix
            CRB_vs_SNR(i_SNR, i_omega, i_N) = 1 / FIM; % CRB 
        end
    end
end


% Plotting
figure;
for i_N = 1:length(N_values)
    for i_omega = 1:length(omega_values)
        subplot(length(N_values), length(omega_values), (i_N-1)*length(omega_values) + i_omega);
        semilogy(SNR_dB, MSE_vs_SNR(:, i_omega, i_N), 'o-', SNR_dB, CRB_vs_SNR(:, i_omega, i_N), 'x-');
        title(['N = ' num2str(N_values(i_N)) ', \omega = ' num2str(omega_values(i_omega, i_N)/pi) '\pi']);
        xlabel('SNR (dB)');
        ylabel('MSE');
        legend('MSE', 'CRB');
        grid on;
    end
end

%% 1.2 Frequency estimation
% 1.2.2 L=2
% Uncorrelated noise 

% Parameters
N = 512;           
omega1 = pi/4;     
% delta_omega_values = 2*pi/N * [2, 6, 32, 64];
delta_omega_values = pi*linspace(1e-2, 3+pi/4, 100);  
SNR_dB_values = -20:5:20;  
num_realizations = 500;   

MSE_omega1 = zeros(length(SNR_dB_values), length(delta_omega_values));
MSE_omega2 = zeros(length(SNR_dB_values), length(delta_omega_values));
CRB_omega1 = zeros(length(SNR_dB_values), length(delta_omega_values));
CRB_omega2 = zeros(length(SNR_dB_values), length(delta_omega_values));

for i = 1:length(SNR_dB_values)
    SNR = 10^(SNR_dB_values(i)/10);
    sigma_w = 1/sqrt(2*SNR); 

    for j = 1:length(delta_omega_values)
        delta_omega = delta_omega_values(j);
        omega2 = omega1 + delta_omega;

          % CRB Calculation
           [CRB_omega1(i,j), CRB_omega2(i,j)] = calculate_CRB(1, 1, omega1, omega2, sigma_w^2);

        for k = 1:num_realizations
            phi1 = 2*pi*rand();
            phi2 = 2*pi*rand();

            n = 0:N-1;
            x = cos(omega1*n + phi1) + cos(omega2*n + phi2) + sigma_w*randn(1, N);

            % Frequency estimation
            Y = abs(fft(x)).^2;
            [~, peak_indices] = maxk(Y(1:N/2+1), 2);
            omega1_hat = (peak_indices(1)-1)*2*pi/N;
            omega2_hat = (peak_indices(2)-1)*2*pi/N;

            MSE_omega1(i, j) = MSE_omega1(i, j) + (omega1_hat - omega1)^2;
            MSE_omega2(i, j) = MSE_omega2(i, j) + (omega2_hat - omega2)^2;

          
        end
    end
end

        MSE_omega1 = MSE_omega1 / num_realizations;
        MSE_omega2 = MSE_omega2 / num_realizations;


% Plotting for omega1
figure;
for j = 1:length(delta_omega_values)
    semilogy(SNR_dB_values, MSE_omega1(:, j), 'o-', 'DisplayName', ['\Delta\omega = ', num2str(delta_omega_values(j))]);
    hold on;
    semilogy(SNR_dB_values, CRB_omega1(:, j), 'x--', 'DisplayName', 'CRB \omega_1');
end

xlabel('SNR (dB)');
ylabel('MSE');
title('MSE vs SNR for \omega_1');
legend;
grid on;

% Plotting for omega2
figure;
for j = 1:length(delta_omega_values)
    semilogy(SNR_dB_values, MSE_omega2(:, j), 'o-', 'DisplayName', ['\Delta\omega = ', num2str(delta_omega_values(j))]);
    hold on;
    semilogy(SNR_dB_values, CRB_omega2(:, j), 'x--', 'DisplayName', 'CRB \omega_2');
end

xlabel('SNR (dB)');
ylabel('MSE');
title('MSE vs SNR for \omega_2');
legend;
grid on;


%% 1.2 Frequency estimation
% 1.2.2 L=2
% Correlated noise 

% Parameters
N = 512;
omega1 = pi/4;     % Frequency of the first sinusoid
%delta_omega_values = 2*pi/N * [2, 6, 32, 64];
delta_omega_values = pi*linspace(1e-2, 3+pi/4, 100); 
SNR_dB_values = -20:5:20; 
num_realizations = 500;   % Number of Monte Carlo realizations
rho = 9/10;

MSE_omega1 = zeros(length(SNR_dB_values), length(delta_omega_values));
MSE_omega2 = zeros(length(SNR_dB_values), length(delta_omega_values));
CRB_omega1 = zeros(size(SNR_dB_values));
CRB_omega2 = zeros(size(SNR_dB_values));

for i = 1:length(SNR_dB_values)
    SNR = 10^(SNR_dB_values(i)/10);
    sigma_w = 1/sqrt(2*SNR); 

    Cww = sigma_w^2 * rho.^(abs((1:N)-(1:N)'));

    for j = 1:length(delta_omega_values)
        delta_omega = delta_omega_values(j);
        omega2 = omega1 + delta_omega;

         % CRB Calculation
           [CRB_omega1(i,j), CRB_omega2(i,j)] = calculate_CRB(1, 1, omega1, omega2, Cww);

        for k = 1:num_realizations
            phi1 = 2*pi*rand();
            phi2 = 2*pi*rand();

            % Generate the signal with correlated noise
            n = 0:N-1;
            noise = mvnrnd(zeros(1, N), Cww);
            x = cos(omega1*n + phi1) + cos(omega2*n + phi2) + noise;

            % Frequency estimation
            Y = abs(fft(x)).^2;
            [~, peak_indices] = maxk(Y(1:N/2+1), 2);
            omega1_hat = (peak_indices(1)-1)*2*pi/N;
            omega2_hat = (peak_indices(2)-1)*2*pi/N;

            MSE_omega1(i, j) = MSE_omega1(i, j) + (omega1_hat - omega1)^2;
            MSE_omega2(i, j) = MSE_omega2(i, j) + (omega2_hat - omega2)^2;

           
        end
    end
end

MSE_omega1 = MSE_omega1 / num_realizations;
MSE_omega2 = MSE_omega2 / num_realizations;

% Plotting for omega1
figure;
for j = 1:length(delta_omega_values)
    semilogy(SNR_dB_values, MSE_omega1(:, j), 'o-', 'DisplayName', ['\Delta\omega = ', num2str(delta_omega_values(j))]);
    hold on;
     semilogy(SNR_dB_values, CRB_omega1(:, j), 'x--', 'DisplayName', 'CRB \omega_1');
end

xlabel('SNR (dB)');
ylabel('MSE');
title('MSE vs SNR for \omega_1 (Correlated Noise)');
legend;
grid on;

% Plotting for omega2
figure;
for j = 1:length(delta_omega_values)
    semilogy(SNR_dB_values, MSE_omega2(:, j), 'o-', 'DisplayName', ['\Delta\omega = ', num2str(delta_omega_values(j))]);
    hold on;
    semilogy(SNR_dB_values, CRB_omega2(:, j), 'x--', 'DisplayName', 'CRB \omega_2');
end

xlabel('SNR (dB)');
ylabel('MSE');
title('MSE vs SNR for \omega_2 (Correlated Noise)');
legend;
grid on;

%%

% CRB Calculation Function (Uncorrelated noise)
% function [CRB1, CRB2] = calculate_CRB(a1, a2, omega1, omega2, sigma2)
%     N = 512;
%     n = 0:N-1;
% 
%     d1_domega1 = -a1*n.*sin(omega1*n);
%     d1_domega2 = zeros(size(n));
%     d2_domega1 = zeros(size(n));
%     d2_domega2 = -a2*n.*sin(omega2*n);
% 
%     FIM = zeros(2, 2);
%     FIM(1, 1) = sum(d1_domega1.^2) / sigma2;
%     FIM(1, 2) = sum(d1_domega1.*d2_domega2) / sigma2;
%     FIM(2, 1) = FIM(1, 2);
%     FIM(2, 2) = sum(d2_domega2.^2) / sigma2;
% 
%     CRB = inv(FIM);
%     CRB1 = CRB(1, 1);
%     CRB2 = CRB(2, 2);
%  end

% CRB Calculation Function (Correlated noise)

function [CRB1, CRB2] = calculate_CRB(a1, a2, omega1, omega2, Cww)
    N = length(Cww);  
    n = 0:N-1;

    d1_domega1 = -a1*n.*sin(omega1*n);
    d1_domega2 = zeros(size(n));
    d2_domega1 = zeros(size(n));
    d2_domega2 = -a2*n.*sin(omega2*n);

    FIM = zeros(2, 2);
    FIM(1, 1) = d1_domega1 * inv(Cww) * d1_domega1';
    FIM(1, 2) = d1_domega1 * inv(Cww) * d2_domega2';
    FIM(2, 1) = FIM(1, 2);
    FIM(2, 2) = d2_domega2 * inv(Cww) * d2_domega2';

    CRB = inv(FIM);
    CRB1 = CRB(1, 1);
    CRB2 = CRB(2, 2);
end
