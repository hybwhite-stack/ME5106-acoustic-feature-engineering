%% Automatically segment the saved 8 kHz-denoised recording (base MATLAB)
% Run this file in MATLAB. Inputs are never edited. Outputs are placed in a
% new folder on each run. Boundaries are acoustic-event estimates, not laser
% trigger timestamps. All samples retain the status NeedsVisualReview.
clear; clc;
rootDir=fileparts(fileparts(mfilename('fullpath')));
rawFile=fullfile(rootDir,'data','raw','Mini Project 3 Data.mat');
denoisedFile=fullfile(rootDir,'data','processed','Mini Project 3 Data_denoised_8kHz.mat');
outDir=fullfile(rootDir,'data','segmented_generated');
if isfolder(outDir)
    outDir=[outDir '_' char(datetime('now','Format','yyyyMMdd_HHmmss'))];
end
mkdir(outDir); mkdir(fullfile(outDir,'review_png'));
mkdir(fullfile(outDir,'review_fig')); mkdir(fullfile(outDir,'contact_sheets'));
S=load(denoisedFile,'IEPE0_Denoised','Fs');
y=S.IEPE0_Denoised; Fs=S.Fs; clear S;
R=load(rawFile,'IEPE0_Raw'); x=R.IEPE0_Raw; clear R;
assert(iscolumn(x)&&iscolumn(y)&&numel(x)==numel(y),'Input size mismatch.');
assert(all(isfinite(x))&&all(isfinite(y)),'Non-finite input values.');
N=numel(y); frameLength=round(Fs*0.001); dt=frameLength/Fs;
nf=floor(N/frameLength);
frameRMS=sqrt(mean(reshape(y(1:nf*frameLength),frameLength,[]).^2,1))';
frameTime=((0:nf-1)'+0.5)*dt;
baseline=median(frameRMS);
spread=median(abs(frameRMS-baseline)); % Unscaled median absolute deviation
highThreshold=baseline+30*spread;
lowThreshold=baseline+6*spread;
minimumCoreFrames=ceil(0.005/dt);
groupGapSeconds=0.100;

% Seed each event using a high threshold; keep sustained events >=5 ms.
[allCoreStart,allCoreEnd]=runs(frameRMS>highThreshold);
keep=(allCoreEnd-allCoreStart+1)>=minimumCoreFrames;
discarded=table((allCoreStart(~keep)-1)*dt,allCoreEnd(~keep)*dt, ...
    (allCoreEnd(~keep)-allCoreStart(~keep)+1)*dt, ...
    'VariableNames',{'Start_s','EndExclusive_s','Duration_s'});
writetable(discarded,fullfile(outDir,'short_candidates_excluded.csv'));
coreStart=allCoreStart(keep); coreEnd=allCoreEnd(keep);
startFrame=coreStart; endFrame=coreEnd;
for k=1:numel(coreStart)
    while startFrame(k)>1 && frameRMS(startFrame(k)-1)>lowThreshold
        startFrame(k)=startFrame(k)-1;
    end
    while endFrame(k)<nf && frameRMS(endFrame(k)+1)>lowThreshold
        endFrame(k)=endFrame(k)+1;
    end
end
assert(all(startFrame(2:end)>endFrame(1:end-1)), ...
    'Expanded events overlap; review thresholds rather than forcing counts.');
gapSeconds=(startFrame(2:end)-endFrame(1:end-1)-1)*dt;
sampleID=cumsum([1;gapSeconds>groupGapSeconds]);
counts=accumarray(sampleID,1); numberGroups=numel(counts); numberTracks=numel(coreStart);
trackID=zeros(numberTracks,1);
for g=1:numberGroups, trackID(sampleID==g)=(1:counts(g))'; end
startIndex=(startFrame-1)*frameLength+1;
endIndex=endFrame*frameLength;
startSeconds=(startIndex-1)/Fs;
endExclusiveSeconds=endIndex/Fs;
coreStartSeconds=(coreStart-1)*dt;
coreEndExclusiveSeconds=coreEnd*dt;
durationSeconds=(endIndex-startIndex+1)/Fs;
status=repmat("NeedsVisualReview",numberTracks,1);
Tracks=table((1:numberTracks)',sampleID,trackID,startIndex,endIndex, ...
    startSeconds,endExclusiveSeconds,durationSeconds,coreStartSeconds, ...
    coreEndExclusiveSeconds,status,'VariableNames', ...
    {'GlobalTrackID','SampleID','TrackID','StartIndex','EndIndex', ...
    'Start_s','EndExclusive_s','Duration_s','CoreStart_s', ...
    'CoreEndExclusive_s','ReviewStatus'});
writetable(Tracks,fullfile(outDir,'track_boundaries_245.csv'));

groupStarts=accumarray(sampleID,startIndex,[],@min);
groupEnds=accumarray(sampleID,endIndex,[],@max);
groupDuration=(groupEnds-groupStarts+1)/Fs;
activeSeconds=accumarray(sampleID,durationSeconds);
nextGap=[(groupStarts(2:end)-groupEnds(1:end-1)-1)/Fs;NaN];
Samples=table((1:numberGroups)',counts,groupStarts,groupEnds, ...
    (groupStarts-1)/Fs,groupEnds/Fs,groupDuration,activeSeconds,nextGap, ...
    counts==5,repmat("NeedsVisualReview",numberGroups,1),'VariableNames', ...
    {'SampleID','TrackCount','StartIndex','EndIndex','Start_s', ...
    'EndExclusive_s','Span_s','TotalTrackDuration_s','GapToNextSample_s', ...
    'FiveTracksCheck','ReviewStatus'});
writetable(Samples,fullfile(outDir,'sample_boundaries_49.csv'));

% Sensitivity: fixed noise baseline, multiple high thresholds; no target-count tuning.
factors=[20;25;30;35;40;50]; sensitivityCounts=zeros(numel(factors),1);
sensitivityGroups=zeros(numel(factors),1);
for j=1:numel(factors)
    [s,t]=runs(frameRMS>(baseline+factors(j)*spread));
    accepted=t-s+1>=minimumCoreFrames; s=s(accepted);t=t(accepted);
    sensitivityCounts(j)=numel(s);
    sensitivityGroups(j)=double(~isempty(s))+sum((s(2:end)-t(1:end-1)-1)*dt>groupGapSeconds);
end
Sensitivity=table(factors,baseline+factors*spread,sensitivityCounts, ...
    sensitivityGroups,'VariableNames', ...
    {'HighThresholdMADMultiplier','HighThreshold','TrackCount','GroupCount'});
writetable(Sensitivity,fullfile(outDir,'threshold_sensitivity.csv'));

% Preserve real numerical track arrays as well as boundaries.
rawTracks=cell(numberGroups,max(counts)); denoisedTracks=rawTracks;
for k=1:numberTracks
    indices=startIndex(k):endIndex(k);
    rawTracks{sampleID(k),trackID(k)}=x(indices);
    denoisedTracks{sampleID(k),trackID(k)}=y(indices);
end
settings=struct('Fs',Fs,'FrameLength',frameLength,'FrameSeconds',dt, ...
    'BaselineMedianRMS',baseline,'UnscaledMAD',spread, ...
    'HighThreshold',highThreshold,'LowThreshold',lowThreshold, ...
    'MinimumCoreSeconds',minimumCoreFrames*dt,'GroupGapSeconds',groupGapSeconds, ...
    'RawSource',rawFile,'DenoisedSource',denoisedFile, ...
    'BoundaryConvention','MATLAB indices inclusive; time intervals [Start_s, EndExclusive_s)', ...
    'MATLABVersion',version,'CreatedAt',char(datetime('now')));
save(fullfile(outDir,'segmented_tracks.mat'),'Tracks','Samples','Sensitivity', ...
    'rawTracks','denoisedTracks','settings','frameTime','frameRMS','-v7.3');
fprintf('Detected %d tracks, %d natural groups; min/max tracks per group %d/%d.\n', ...
    numberTracks,numberGroups,min(counts),max(counts));
disp(Sensitivity);
if numberGroups~=49 || numberTracks~=245 || any(counts~=5)
    warning('Structure differs from assignment. Outputs preserve the actual detections.');
end

% Figures: exact samples in every group, no waveform downsampling.
purple=[0.49 0.18 0.56]; green=[0 0.45 0.20];
plotSegments=cell(numberGroups,1); contextSeconds=0.050;
for g=1:numberGroups
    plotStart=max(1,groupStarts(g)-round(contextSeconds*Fs));
    plotEnd=min(N,groupEnds(g)+round(contextSeconds*Fs));
    idx=(plotStart:plotEnd)'; tt=(idx-1)/Fs;
    plotSegments{g}=struct('t',tt,'raw',x(idx),'denoised',y(idx));
    fig=figure('Visible','off','Color','w','Position',[60 60 1450 950]);
    tl=tiledlayout(fig,3,1,'Padding','compact','TileSpacing','compact');
    a1=nexttile(tl);
    plot(a1,tt,x(idx),'--','Color',purple,'DisplayName','Raw');hold(a1,'on');
    plot(a1,tt,y(idx),'-','Color',green,'DisplayName','Denoised 8 kHz');
    ylabel(a1,'Amplitude (raw units)');title(a1,'Raw and denoised waveforms');
    legend(a1,'Location','northeast');grid(a1,'on');
    a2=nexttile(tl);
    plot(a2,tt,y(idx),'-','Color',green,'DisplayName','Denoised');hold(a2,'on');
    bound=max(abs(y(idx)))*1.16; if bound==0,bound=1;end
    ylim(a2,[-bound bound]);ylabel(a2,'Amplitude (raw units)');
    title(a2,'Acoustic boundaries: dashed=start; solid=end');grid(a2,'on');
    selected=find(sampleID==g);
    for k=selected'
        for ax=[a1 a2]
            xline(ax,startSeconds(k),'--k','HandleVisibility','off');
            xline(ax,endExclusiveSeconds(k),'-k','HandleVisibility','off');
        end
        text(a2,(startSeconds(k)+endExclusiveSeconds(k))/2,bound*0.89, ...
            sprintf('T%d',trackID(k)),'HorizontalAlignment','center','FontWeight','bold');
    end
    a3=nexttile(tl);
    fr=frameTime>=tt(1)&frameTime<=tt(end);
    semilogy(a3,frameTime(fr),frameRMS(fr),'Color',green,'DisplayName','1 ms RMS');
    hold(a3,'on');
    yline(a3,highThreshold,'--','Color',purple,'DisplayName','High threshold (seed)');
    yline(a3,lowThreshold,':k','DisplayName','Low threshold (boundary)');
    grid(a3,'on');ylabel(a3,'RMS (log scale)');xlabel(a3,'Time from recording start (s)');
    legend(a3,'Location','northeast');
    for k=selected'
        xline(a3,startSeconds(k),'--k','HandleVisibility','off');
        xline(a3,endExclusiveSeconds(k),'-k','HandleVisibility','off');
    end
    linkaxes([a1 a2 a3],'x');xlim(a1,[tt(1) tt(end)]);
    title(tl,sprintf('Sample %02d: %d tracks | 1 ms boundary resolution | visual review pending',g,counts(g)));
    fileStem=sprintf('sample_%02d_review',g);
    savefig(fig,fullfile(outDir,'review_fig',[fileStem '.fig']));
    exportgraphics(fig,fullfile(outDir,'review_png',[fileStem '.png']),'Resolution',140);
    close(fig);
    fprintf('Exported review %02d/%02d.\n',g,numberGroups);
end

% Contact sheets use the same amplitude limits across every group.
commonY=max(cellfun(@(v)max(abs(v.denoised)),plotSegments))*1.12;
perSheet=7; sheets=ceil(numberGroups/perSheet);
for sh=1:sheets
    first=(sh-1)*perSheet+1; last=min(sh*perSheet,numberGroups);
    fig=figure('Visible','off','Color','w','Position',[60 60 1400 1600]);
    tl=tiledlayout(fig,last-first+1,1,'Padding','compact','TileSpacing','compact');
    for g=first:last
        ax=nexttile(tl); v=plotSegments{g}; relative=v.t-(groupStarts(g)-1)/Fs;
        plot(ax,relative,v.denoised,'Color',green);hold(ax,'on');
        ylim(ax,[-commonY commonY]);grid(ax,'on');
        xlim(ax,[relative(1) relative(end)]);
        selected=find(sampleID==g);
        for k=selected'
            xline(ax,startSeconds(k)-(groupStarts(g)-1)/Fs,'--k','HandleVisibility','off');
            xline(ax,endExclusiveSeconds(k)-(groupStarts(g)-1)/Fs,'-k','HandleVisibility','off');
        end
        title(ax,sprintf('Sample %02d | %d tracks | recording %.3f-%.3f s', ...
            g,counts(g),(groupStarts(g)-1)/Fs,groupEnds(g)/Fs));
        ylabel(ax,'Amplitude');
    end
    xlabel(tl,'Time relative to sample start (s)');
    title(tl,sprintf('Samples %02d-%02d | common amplitude scale | dashed=start, solid=end',first,last));
    exportgraphics(fig,fullfile(outDir,'contact_sheets',sprintf('contact_%02d.png',sh)), ...
        'Resolution',120);close(fig);
end

fig=figure('Visible','off','Color','w','Position',[60 60 1500 650]);
semilogy(frameTime,frameRMS,'Color',green);hold on;
yline(highThreshold,'--','High threshold','Color',purple);
yline(lowThreshold,':k','Low threshold');
for g=1:numberGroups
    text((groupStarts(g)-1)/Fs,max(frameRMS)*1.6,sprintf('%d',g), ...
        'HorizontalAlignment','center','FontSize',7,'Rotation',90);
end
ylim([min(frameRMS)*0.7 max(frameRMS)*4]);xlim([0 N/Fs]);grid on;
xlabel('Time (s)');ylabel('1 ms RMS (log scale)');
title(sprintf('Full recording: %d natural groups, %d retained tracks',numberGroups,numberTracks));
savefig(fig,fullfile(outDir,'segmentation_overview.fig'));
exportgraphics(fig,fullfile(outDir,'segmentation_overview.png'),'Resolution',140);close(fig);
assert(all(Tracks.EndIndex>=Tracks.StartIndex));
assert(all(Tracks.StartIndex(2:end)>Tracks.EndIndex(1:end-1)));
assert(all(cellfun(@numel,rawTracks(:))==cellfun(@numel,denoisedTracks(:))));
fprintf('Track duration range %.3f-%.3f s.\n',min(durationSeconds),max(durationSeconds));
fprintf('Within-group gap range %.3f-%.3f s.\n', ...
    min(gapSeconds(gapSeconds<=groupGapSeconds)),max(gapSeconds(gapSeconds<=groupGapSeconds)));
fprintf('Between-group gap range %.3f-%.3f s.\n',min(nextGap),max(nextGap));
fprintf('Short candidates excluded: %d.\n',height(discarded));
fprintf('Outputs: %s\n',outDir);

function [s,t]=runs(mask)
    d=diff([false;mask(:);false]);s=find(d==1);t=find(d==-1)-1;
end

