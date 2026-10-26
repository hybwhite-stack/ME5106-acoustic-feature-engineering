%% Denoise Mini Project 3 acoustic data and compare with the raw signal
% Method: remove the lowest 10% of the one-sided frequency range.
% Fs = 160 kHz, Nyquist = 80 kHz, so the nominal cutoff is 8 kHz.
%
% This script uses only base MATLAB. The original MAT file is never changed.

clear;
clc;

rootDir = fileparts(fileparts(mfilename('fullpath')));
inputFile = fullfile(rootDir, 'data', 'raw', 'Mini Project 3 Data.mat');
outputFile = fullfile(rootDir, 'data', 'processed', 'Mini Project 3 Data_denoised_8kHz.mat');
figureFile = fullfile(rootDir, 'figures', '原始与8kHz高通对比.png');

Fs = 160000;
cutoffHz = 0.10 * (Fs / 2);  % 8 kHz
filterOrder = 800;           % Even order; 801-tap linear-phase FIR
frameLength = round(0.010 * Fs);  % 10 ms for the overview RMS

if ~isfile(inputFile)
    error('Input file not found: %s', inputFile);
end
if isfile(outputFile)
    error(['Output already exists. Rename or remove it before rerunning: ' ...
        '%s'], outputFile);
end

source = matfile(inputFile);
info = whos(source, 'IEPE0_Raw');
if isempty(info) || numel(info.size) ~= 2 || info.size(2) ~= 1
    error('IEPE0_Raw was not found as a column vector.');
end
numberSamples = info.size(1);

%% Design a windowed-sinc high-pass FIR without toolbox functions
halfOrder = filterOrder / 2;
tapIndex = -halfOrder:halfOrder;
normalizedArgument = 2 * cutoffHz / Fs * tapIndex;
sincValues = ones(size(normalizedArgument));
nonzero = normalizedArgument ~= 0;
sincValues(nonzero) = sin(pi * normalizedArgument(nonzero)) ./ ...
    (pi * normalizedArgument(nonzero));

lowpass = (2 * cutoffHz / Fs) * sincValues;
hammingWindow = 0.54 - 0.46 * cos(2 * pi * (0:filterOrder) / filterOrder);
lowpass = lowpass .* hammingWindow;
lowpass = lowpass / sum(lowpass);

highpass = -lowpass;
highpass(halfOrder + 1) = highpass(halfOrder + 1) + 1;

% Normalize the gain at the Nyquist frequency to one.
nyquistResponse = sum(highpass .* ((-1) .^ (0:filterOrder)));
highpass = highpass / abs(nyquistResponse);
highpass = highpass(:);
groupDelay = halfOrder;

%% Apply the FIR by FFT overlap-save and write the output incrementally
destination = matfile(outputFile, 'Writable', true);
destination.IEPE0_Denoised(numberSamples, 1) = 0;
destination.Fs = Fs;
destination.CutoffHz = cutoffHz;
destination.FilterOrder = filterOrder;
destination.FilterCoefficients = highpass;
destination.ProcessingNote = [ ...
    'Linear-phase FIR high-pass. Nominal cutoff 8 kHz, interpreted from ' ...
    'the lowest 10 percent of the 0-80 kHz one-sided frequency range.'];

filterLength = numel(highpass);
fftLength = 2^20;
validBlockLength = fftLength - filterLength + 1;
filterSpectrum = fft(highpass, fftLength);
previousInput = zeros(filterLength - 1, 1);

inputStart = 1;
while inputStart <= numberSamples
    samplesThisBlock = min(validBlockLength, numberSamples - inputStart + 1);
    inputBlock = source.IEPE0_Raw( ...
        inputStart:(inputStart + samplesThisBlock - 1), 1);

    paddedBlock = [previousInput; inputBlock; ...
        zeros(validBlockLength - samplesThisBlock, 1)];
    filteredBlock = real(ifft(fft(paddedBlock) .* filterSpectrum));
    causalOutput = filteredBlock( ...
        filterLength:(filterLength + samplesThisBlock - 1));

    causalStart = inputStart;
    causalEnd = inputStart + samplesThisBlock - 1;
    alignedStart = max(1, causalStart - groupDelay);
    alignedEnd = min(numberSamples, causalEnd - groupDelay);

    if alignedEnd >= alignedStart
        sourceOffset = alignedStart + groupDelay - causalStart + 1;
        count = alignedEnd - alignedStart + 1;
        destination.IEPE0_Denoised(alignedStart:alignedEnd, 1) = ...
            causalOutput(sourceOffset:(sourceOffset + count - 1));
    end

    if samplesThisBlock >= filterLength - 1
        previousInput = inputBlock((end - filterLength + 2):end);
    else
        previousInput = [previousInput((samplesThisBlock + 1):end); inputBlock];
    end

    inputStart = inputStart + samplesThisBlock;
    fprintf('Filtered %.1f%% of the recording.\n', ...
        100 * min(inputStart - 1, numberSamples) / numberSamples);
end

% Complete the group-delay compensation at the end using zero padding.
tailBlock = [previousInput; zeros(groupDelay, 1); ...
    zeros(validBlockLength - groupDelay, 1)];
tailFiltered = real(ifft(fft(tailBlock) .* filterSpectrum));
tailCausal = tailFiltered(filterLength:(filterLength + groupDelay - 1));
destination.IEPE0_Denoised((numberSamples - groupDelay + 1):numberSamples, 1) = ...
    tailCausal;

%% Compute 10 ms RMS for the full-recording overview without loading it all
numberFrames = floor(numberSamples / frameLength);
rawRms = zeros(numberFrames, 1);
denoisedRms = zeros(numberFrames, 1);
framesPerBatch = 200;

for firstFrame = 1:framesPerBatch:numberFrames
    lastFrame = min(numberFrames, firstFrame + framesPerBatch - 1);
    firstSample = (firstFrame - 1) * frameLength + 1;
    lastSample = lastFrame * frameLength;

    rawBatch = source.IEPE0_Raw(firstSample:lastSample, 1);
    denoisedBatch = destination.IEPE0_Denoised(firstSample:lastSample, 1);
    rawBatch = reshape(rawBatch, frameLength, []);
    denoisedBatch = reshape(denoisedBatch, frameLength, []);

    rawBatch = rawBatch - mean(rawBatch, 1);
    denoisedBatch = denoisedBatch - mean(denoisedBatch, 1);
    frameRange = firstFrame:lastFrame;
    rawRms(frameRange) = sqrt(mean(rawBatch.^2, 1));
    denoisedRms(frameRange) = sqrt(mean(denoisedBatch.^2, 1));
end

frameTime = (((1:numberFrames)' - 0.5) * frameLength) / Fs;

%% Select a representative region containing Sample 1 for waveform comparison
zoomStartSeconds = 6.20;
zoomEndSeconds = 6.55;
zoomStart = floor(zoomStartSeconds * Fs) + 1;
zoomEnd = min(numberSamples, floor(zoomEndSeconds * Fs));
rawZoom = source.IEPE0_Raw(zoomStart:zoomEnd, 1);
denoisedZoom = destination.IEPE0_Denoised(zoomStart:zoomEnd, 1);
zoomTime = ((zoomStart:zoomEnd)' - 1) / Fs;

%% Calculate comparable one-sided spectra for the representative region
spectrumLength = numel(rawZoom);
windowIndex = (0:(spectrumLength - 1))';
hannWindow = 0.5 - 0.5 * cos(2 * pi * windowIndex / ...
    max(spectrumLength - 1, 1));
fftPoints = 2^nextpow2(spectrumLength);

rawWindowed = (rawZoom - mean(rawZoom)) .* hannWindow;
denoisedWindowed = (denoisedZoom - mean(denoisedZoom)) .* hannWindow;
rawSpectrum = fft(rawWindowed, fftPoints);
denoisedSpectrum = fft(denoisedWindowed, fftPoints);
frequency = (0:(fftPoints / 2))' * Fs / fftPoints;

windowEnergy = sum(hannWindow.^2);
rawPsd = abs(rawSpectrum(1:(fftPoints / 2 + 1))).^2 / ...
    (Fs * windowEnergy);
denoisedPsd = abs(denoisedSpectrum(1:(fftPoints / 2 + 1))).^2 / ...
    (Fs * windowEnergy);
rawPsd(2:end-1) = 2 * rawPsd(2:end-1);
denoisedPsd(2:end-1) = 2 * denoisedPsd(2:end-1);

lowBand = frequency < cutoffHz;
highBand = frequency >= cutoffHz;
rawLowPower = sum(rawPsd(lowBand)) * (Fs / fftPoints);
denoisedLowPower = sum(denoisedPsd(lowBand)) * (Fs / fftPoints);
rawHighPower = sum(rawPsd(highBand)) * (Fs / fftPoints);
denoisedHighPower = sum(denoisedPsd(highBand)) * (Fs / fftPoints);
lowBandAttenuationDb = 10 * log10(denoisedLowPower / rawLowPower);
highBandChangeDb = 10 * log10(denoisedHighPower / rawHighPower);

%% Create an accessible comparison figure
purple = [0.49, 0.18, 0.56];
green = [0.00, 0.45, 0.20];

fig = figure('Visible', 'off', 'Color', 'w', ...
    'Position', [100, 100, 1500, 1100]);
layout = tiledlayout(fig, 3, 1, 'TileSpacing', 'compact', ...
    'Padding', 'compact');

ax1 = nexttile(layout);
plot(ax1, frameTime, rawRms, '--', 'Color', purple, 'LineWidth', 0.9, ...
    'DisplayName', 'Raw signal');
hold(ax1, 'on');
plot(ax1, frameTime, denoisedRms, '-', 'Color', green, 'LineWidth', 1.0, ...
    'DisplayName', 'Denoised signal');
grid(ax1, 'on');
xlabel(ax1, 'Time (s)');
ylabel(ax1, '10 ms RMS amplitude');
title(ax1, 'Full recording: short-time RMS');
legend(ax1, 'Location', 'northeast');

ax2 = nexttile(layout);
plot(ax2, zoomTime, rawZoom, '--', 'Color', purple, 'LineWidth', 0.75, ...
    'DisplayName', 'Raw signal');
hold(ax2, 'on');
plot(ax2, zoomTime, denoisedZoom, '-', 'Color', green, 'LineWidth', 0.85, ...
    'DisplayName', 'Denoised signal');
grid(ax2, 'on');
xlabel(ax2, 'Time (s)');
ylabel(ax2, 'Amplitude');
title(ax2, 'Waveform zoom: region containing Sample 1');
legend(ax2, 'Location', 'best');

ax3 = nexttile(layout);
plot(ax3, frequency / 1000, 10 * log10(rawPsd + eps), '--', ...
    'Color', purple, 'LineWidth', 0.9, 'DisplayName', 'Raw signal');
hold(ax3, 'on');
plot(ax3, frequency / 1000, 10 * log10(denoisedPsd + eps), '-', ...
    'Color', green, 'LineWidth', 1.0, 'DisplayName', 'Denoised signal');
xline(ax3, cutoffHz / 1000, ':k', '8 kHz cutoff', ...
    'LabelVerticalAlignment', 'bottom', 'HandleVisibility', 'off');
grid(ax3, 'on');
xlim(ax3, [0, Fs / 2000]);
xlabel(ax3, 'Frequency (kHz)');
ylabel(ax3, 'Power spectral density (dB/Hz)');
title(ax3, 'Frequency-domain verification');
legend(ax3, 'Location', 'best');

title(layout, sprintf(['Raw (purple dashed) vs denoised (green solid): ' ...
    '8 kHz high-pass FIR']));
exportgraphics(fig, figureFile, 'Resolution', 180);
close(fig);

fprintf('\nCompleted.\n');
fprintf('Input:  %s\n', inputFile);
fprintf('Output: %s\n', outputFile);
fprintf('Figure: %s\n', figureFile);
fprintf('Samples: %d; duration: %.3f s\n', numberSamples, numberSamples / Fs);
fprintf('Nominal cutoff: %.1f Hz\n', cutoffHz);
fprintf('Low-band power change (0-8 kHz): %.2f dB\n', lowBandAttenuationDb);
fprintf('High-band power change (8-80 kHz): %.2f dB\n', highBandChangeDb);


