%% Calculate time and frequency features from the approved 245 track slices.
% Base MATLAB only. Reads the saved boundaries and denoised signal slices;
% does not redetect boundaries or modify the source waveforms.
clear; clc;
rootDir=fileparts(fileparts(mfilename('fullpath')));
outDir=fullfile(rootDir,'data','features');
sourceFile=fullfile(rootDir,'data','segmented','分割数据_245段.mat');
if ~isfile(sourceFile), error('Missing segmented tracks: %s',sourceFile); end
S=load(sourceFile,'Tracks','Samples','denoisedTracks','settings');
Fs=S.settings.Fs;
assert(Fs==160000 && height(S.Tracks)==245 && height(S.Samples)==49);
assert(isequal(size(S.denoisedTracks),[49 5]));
assert(all(S.Samples.TrackCount==5));

TrackFeatures=S.Tracks(:,{'GlobalTrackID','SampleID','TrackID', ...
    'StartIndex','EndIndex','Start_s','EndExclusive_s','Duration_s'});
nTracks=height(TrackFeatures);
RMS=zeros(nTracks,1);
PeakAbs=zeros(nTracks,1);
PeakToPeak=zeros(nTracks,1);
SignalEnergy_Amp2_s=zeros(nTracks,1);
CrestFactor=zeros(nTracks,1);
DominantFrequency_Hz=zeros(nTracks,1);
SpectralCentroid_Hz=zeros(nTracks,1);
Band8to20k_Fraction=zeros(nTracks,1);
Band20to40k_Fraction=zeros(nTracks,1);
Band40to80k_Fraction=zeros(nTracks,1);
sampleCounts=zeros(49,1);
sampleSquaredSums=zeros(49,1);

for k=1:nTracks
    g=S.Tracks.SampleID(k); j=S.Tracks.TrackID(k);
    x=S.denoisedTracks{g,j};
    assert(iscolumn(x) && all(isfinite(x)) && ...
        numel(x)==S.Tracks.EndIndex(k)-S.Tracks.StartIndex(k)+1);
    n=numel(x);
    squaredSum=sum(x.^2);
    RMS(k)=sqrt(squaredSum/n);
    PeakAbs(k)=max(abs(x));
    PeakToPeak(k)=max(x)-min(x);
    SignalEnergy_Amp2_s(k)=squaredSum/Fs;
    CrestFactor(k)=PeakAbs(k)/RMS(k);
    sampleCounts(g)=sampleCounts(g)+n;
    sampleSquaredSums(g)=sampleSquaredSums(g)+squaredSum;

    % Periodic Hann window, one-sided power. Remove the segment mean only
    % for spectral features to avoid leakage at zero frequency.
    centered=x-mean(x);
    window=0.5-0.5*cos(2*pi*(0:n-1)'/n);
    fftValues=fft(centered.*window);
    p=abs(fftValues(1:floor(n/2)+1)).^2;
    if mod(n,2)==0
        p(2:end-1)=2*p(2:end-1);
    else
        p(2:end)=2*p(2:end);
    end
    frequency=(0:floor(n/2))'*Fs/n;
    pass=frequency>=8000 & frequency<=80000;
    f=frequency(pass); power=p(pass);
    totalPower=sum(power);
    assert(totalPower>0,'Zero spectral power in Track %d.',k);
    [~,imax]=max(power);
    DominantFrequency_Hz(k)=f(imax);
    SpectralCentroid_Hz(k)=sum(f.*power)/totalPower;
    Band8to20k_Fraction(k)=sum(power(f>=8000 & f<20000))/totalPower;
    Band20to40k_Fraction(k)=sum(power(f>=20000 & f<40000))/totalPower;
    Band40to80k_Fraction(k)=sum(power(f>=40000 & f<=80000))/totalPower;
end

TrackFeatures.RMS=RMS;
TrackFeatures.PeakAbs=PeakAbs;
TrackFeatures.PeakToPeak=PeakToPeak;
TrackFeatures.SignalEnergy_Amp2_s=SignalEnergy_Amp2_s;
TrackFeatures.CrestFactor=CrestFactor;
TrackFeatures.DominantFrequency_Hz=DominantFrequency_Hz;
TrackFeatures.SpectralCentroid_Hz=SpectralCentroid_Hz;
TrackFeatures.Band8to20k_Fraction=Band8to20k_Fraction;
TrackFeatures.Band20to40k_Fraction=Band20to40k_Fraction;
TrackFeatures.Band40to80k_Fraction=Band40to80k_Fraction;

% The sample is the experimental unit; summarize its five tracks.
SampleFeatures=S.Samples(:,{'SampleID','TrackCount','Start_s', ...
    'EndExclusive_s','Span_s','TotalTrackDuration_s'});
SampleFeatures.PooledRMS=sqrt(sampleSquaredSums./sampleCounts);
SampleFeatures.TotalSignalEnergy_Amp2_s=zeros(49,1);
metrics={'RMS','PeakAbs','PeakToPeak','SignalEnergy_Amp2_s', ...
    'CrestFactor','DominantFrequency_Hz','SpectralCentroid_Hz', ...
    'Band8to20k_Fraction','Band20to40k_Fraction', ...
    'Band40to80k_Fraction','Duration_s'};
for g=1:49
    rows=TrackFeatures.SampleID==g;
    assert(nnz(rows)==5);
    SampleFeatures.TotalSignalEnergy_Amp2_s(g)= ...
        sum(TrackFeatures.SignalEnergy_Amp2_s(rows));
    for j=1:numel(metrics)
        v=TrackFeatures.(metrics{j})(rows);
        SampleFeatures.([metrics{j} '_Mean'])(g)=mean(v);
        SampleFeatures.([metrics{j} '_SD'])(g)=std(v,0); % Sample SD, n=5.
    end
end

fractions=Band8to20k_Fraction+Band20to40k_Fraction+Band40to80k_Fraction;
assert(all(isfinite(TrackFeatures{:,9:end}),'all'));
assert(all(abs(fractions-1)<1e-10));
assert(all(PeakAbs>=RMS) && all(PeakToPeak>=PeakAbs));
assert(all(DominantFrequency_Hz>=8000 & DominantFrequency_Hz<=80000));
assert(all(SpectralCentroid_Hz>=8000 & SpectralCentroid_Hz<=80000));
assert(all(isfinite(SampleFeatures{:,7:end}),'all'));

Method=struct('Signal','IEPE0_Denoised', ...
    'InputFile',sourceFile,'Fs',Fs,'CutoffHz',8000, ...
    'TimeDomain','Exact saved denoised samples; no additional processing', ...
    'Spectrum','Remove each track mean; periodic Hann; one-sided FFT power', ...
    'SpectralRangeHz',[8000 80000], ...
    'BandsHz',[8000 20000;20000 40000;40000 80000], ...
    'BandConvention','[8000,20000), [20000,40000), [40000,80000]', ...
    'GroupSummary','Mean and sample SD across 5 tracks; pooled RMS uses all track samples');

writetable(TrackFeatures,fullfile(outDir,'track_features_245.csv'));
writetable(SampleFeatures,fullfile(outDir,'sample_features_49.csv'));
save(fullfile(outDir,'features_8kHz.mat'), ...
    'TrackFeatures','SampleFeatures','Method');
fprintf('Saved %d track rows and %d sample rows.\n', ...
    height(TrackFeatures),height(SampleFeatures));
fprintf('RMS range %.6g to %.6g; peak range %.6g to %.6g.\n', ...
    min(RMS),max(RMS),min(PeakAbs),max(PeakAbs));
fprintf('Band-fraction sum max error: %.3g.\n',max(abs(fractions-1)));

