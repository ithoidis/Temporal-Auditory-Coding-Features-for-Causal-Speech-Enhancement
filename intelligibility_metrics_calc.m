clear all;
addpath('C:\Users\VRYSIS\Documents\MATLAB\HASPI_HASQI');
addpath('C:\Users\VRYSIS\Documents\MATLAB\audio');
addpath('C:\Users\VRYSIS\Documents\MATLAB\stoi');
% pkg load signal

%% init 
audiograms = [0 0 0 55 80 90;
              0 15 30 60 80 85;
              14 14 11 14 24 39;
              24 24 25 31 46 60;
              40 40 50 60 65 65;
              45 55 50 30 20 20];
clean_f = dir('C:\Users\VRYSIS\Documents\MATLAB\audio\*_clean.wav');
noisy_f = dir('C:\Users\VRYSIS\Documents\MATLAB\audio\*_noisy.wav');
predi_f = dir('C:\Users\VRYSIS\Documents\MATLAB\audio\*_predi.wav');
assert(length(clean_f) == length(noisy_f));
assert(length(clean_f) == length(predi_f));
n = (length(clean_f));
stoi_results = zeros(n,4);
ha_results = zeros(length(audiograms), n, 4);
snrs = strings(n,1);
noises = strings(n,1);
fs = 16000;

%% Load & Calculate Intelligibility
for i = 1:n
    fprintf('%d/%d - %s \n', i, n, clean_f(i).name)
    clean = audioread(clean_f(i).name);
    noisy = audioread(noisy_f(i).name);
    predi = audioread(predi_f(i).name);
    temp = strsplit(clean_f(i).name, '_');
    snrs(i) = cell2mat(erase(temp(3), 'dB'));
    noises(i) = cell2mat(temp(2));
    stoi_results(i, 1) = stoi(clean, noisy, fs); % stoi_orig
    stoi_results(i, 2) = stoi(clean, predi, fs); % stoi_pred
    stoi_results(i, 3) = estoi(clean, noisy, fs); % estoi_orig
    stoi_results(i, 4) = estoi(clean, predi, fs); % estoi_pred
    fprintf('%.3f ', stoi_results(i,:));
    fprintf('\n');
    
    % adjust gain/normalize
    predi = predi ./ rms(predi);
    clean = clean ./ rms(clean);
    noisy = noisy ./ rms(noisy);

    for k = 1:length(audiograms(:,1))
        ha_results(k, i, 1) = HASPI_v1(clean, fs, noisy, fs, audiograms(k,:), 65);
        ha_results(k, i, 2) = HASPI_v1(clean, fs, predi, fs, audiograms(k,:), 65);
        ha_results(k, i, 3) = HASQI_v2(clean, fs, noisy, fs, audiograms(k,:), 1, 65);
        ha_results(k, i, 4) = HASQI_v2(clean, fs, predi, fs, audiograms(k,:), 1, 65);
        fprintf('%f ', ha_results(k, i, :));
        fprintf('\n');
    end
end

%% Save results to excel
for k = 1: length(audiograms(:,1))
    aud_results = squeeze(ha_results(k,:,:));
    fnames = struct2table(clean_f);
    fnames = fnames(:,1);
    out = [fnames,  array2table(noises),array2table(snrs), array2table(aud_results,'VariableNames',{'HASPI_orig','HASPI_predi','HASQI_orig', 'HASqI_predi'})];
    writetable(out, 'HA_'+ string(k)+'.xls')
end
csvwrite('results_matlab.csv', stoi_results);
desc = 'STOI: orig - pred || eSTOI: orig - pred';
total = sum(stoi_results, 1) ./ n;
desc = 'HASPI: orig - pred || HASQI: orig - pred';
total = squeeze(sum(ha_results, 2)) ./ n;