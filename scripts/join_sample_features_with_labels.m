%% Join slide 3 process-parameter labels to the existing 49-row feature table.
% The unknown targets are NaN, never guessed. Slide 4 states a conflicting
% power for sample 49; slide 3 is the source for the prediction task.
clear; clc;
rootDir=fileparts(fileparts(mfilename('fullpath')));
featuresDir=fullfile(rootDir,'data','features');
featureFile=fullfile(featuresDir,'sample_features_49.csv');
F=readtable(featureFile);
assert(height(F)==49 && numel(unique(F.SampleID))==49);
assert(isequal(sort(F.SampleID),(1:49)'));

LaserPower_W=nan(49,1);
ScanSpeed_mm_s=nan(49,1);

% First three alternating blocks in the slide 3 grid.
LaserPower_W(1:18)=225;
ScanSpeed_mm_s([1:3 7:9 13:15])=850;
ScanSpeed_mm_s([4:6 10:12 16:18])=550;

% Middle six blocks: three 225 W and three 175 W triplets, all 700 mm/s.
LaserPower_W([19:21 25:27 31:33])=225;
LaserPower_W([22:24 28:30 34:36])=175;
ScanSpeed_mm_s(19:36)=700;

% Rightmost seven-column block: read by matching sample ID to grid cell.
LaserPower_W([37 38 39 40 42 44 45 46 47 48])= ...
    [175 275 325 275 370 425 175 275 325 370];
ScanSpeed_mm_s([37 38 39 40 42 44 45 46 47 48 49])= ...
    [550 550 550 700 700 700 850 850 850 850 850];
% Samples 41 and 43: both parameters unknown on slide 3.
% Sample 49: power unknown, speed 850 mm/s on slide 3.

SampleID=(1:49)';
PowerKnown=isfinite(LaserPower_W);
SpeedKnown=isfinite(ScanSpeed_mm_s);
FullyLabeled=PowerKnown & SpeedKnown;
Labels=table(SampleID,LaserPower_W,ScanSpeed_mm_s,PowerKnown, ...
    SpeedKnown,FullyLabeled);
LabeledFeatures=innerjoin(Labels,F,'Keys','SampleID');
LabeledFeatures=sortrows(LabeledFeatures,'SampleID');
assert(height(LabeledFeatures)==49);
assert(isequal(LabeledFeatures.SampleID,SampleID));
assert(nnz(PowerKnown)==46 && nnz(SpeedKnown)==47 && nnz(FullyLabeled)==46);
assert(all(isnan(LaserPower_W([41 43 49]))));
assert(all(isnan(ScanSpeed_mm_s([41 43]))));
assert(ScanSpeed_mm_s(49)==850);
assert(LaserPower_W(1)==225 && ScanSpeed_mm_s(1)==850);
assert(LaserPower_W(44)==425 && ScanSpeed_mm_s(44)==700);

writetable(Labels,fullfile(featuresDir,'process_parameter_labels_49.csv'));
writetable(LabeledFeatures, ...
    fullfile(featuresDir,'sample_features_labeled_49.csv'));
save(fullfile(featuresDir,'labeled_features_8kHz.mat'), ...
    'Labels','LabeledFeatures');
fprintf('Joined %d samples by SampleID. Known power: %d, known speed: %d, both: %d.\n', ...
    height(LabeledFeatures),nnz(PowerKnown),nnz(SpeedKnown),nnz(FullyLabeled));
disp(LabeledFeatures(ismember(LabeledFeatures.SampleID,[41 43 49]), ...
    {'SampleID','LaserPower_W','ScanSpeed_mm_s','PowerKnown','SpeedKnown','FullyLabeled','RMS_Mean'}));

