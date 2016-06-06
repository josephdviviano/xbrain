% Example uses of functions in isc_nifti_kit
% j chen 06/24/13
% j vivs 03/30/15 haxx

% All Nifti files must be unzipped
% Always run this section before running any of the other sections
clear

basepath = '/projects/jdv/code/isc_nifti_kit/'; 
addpath(basepath); 
addpath(fullfile(basepath,'NIFTI_tools')); 

%% ROI Timecourses
% input: 4d functional nifti files for subj01, subj02, subj03
% output: a summary file roi_condname_roicorr.mat that contains ISC info for the ROI

x = 0;
x=x+1; rois{x} = 'pieman_a1_3mm.nii';
x=x+1; rois{x} = 'post_cing_precun_3mm.nii';
for rr = 1:length(rois)
    roi = rois{rr};
    % set roidata load/save flag:
    % 'savedata': run this the first time, after which a .mat will be saved
    % 'loaddata': uses the .mat (faster)
    % if you change the ROI, you need to run with the savedata flag again.
    roidataflag = 'savedata';
    condname = 'PiemanIntact';
    funcnames = []; roinames = [];
    funcnames{1} = fullfile(basepath,'subjects','subj01',condname,'trans_filtered_func_data.nii');
    funcnames{2} = fullfile(basepath,'subjects','subj02',condname,'trans_filtered_func_data.nii');
    funcnames{3} = fullfile(basepath,'subjects','subj03',condname,'trans_filtered_func_data.nii');
    funcnames{4} = fullfile(basepath,'subjects','subj04',condname,'trans_filtered_func_data.nii');
    roinames = fullfile(basepath,'standard',roi);
    fprintf(['cond = ' condname ', output will be saved as ' rois{rr}(1:end-4) '_' condname '_roicorr.mat\n']);
    opts.outputPath = fullfile(basepath,'intersubj','roicorr'); % create this folder in advance
    opts.outputName = condname;
    opts.crop_beginning = 10; % number of TRs to crop from beginning
    opts.crop_end = 10; % number of TRs to crop from end
    opts.crop_special = [1 6 14; 2 12 8; 3 11 9]; % specify different crops for different subjects, otherwise defaults to crop_beginning and crop_end
    opts.roidata = roidataflag;
    nkit_nifti_roi_timecourse(funcnames, roinames, opts);
    if strcmp(roidataflag,'savedata') % if using savedata flag, auto-run loaddata afterward
        opts.roidata = 'loaddata';
        nkit_nifti_roi_timecourse(funcnames, roinames, opts);
    end
end


%% plot ROI timecourses

roicorrpath = fullfile(basepath,'intersubj','roicorr'); 
roicorr_file = 'pieman_a1_3mm_PiemanIntact_roicorr.mat';
r = load(fullfile(roicorrpath,roicorr_file));
figure(9090); clf; set(gcf,'Color',[1 1 1]); rcolors = 'rgbm';
for n = 1:length(r.roicorr), plot(r.roitc(:,n),rcolors(n),'LineWidth',2); hold on; end
plot(zscore(r.meantc),'k','LineWidth',3);
set(gca,'FontSize',16); grid on; xlabel('TR'); ylabel('Z')
rtitle = roicorr_file; rtitle(rtitle=='_')='-';
title(rtitle);

% plot lagged corrs for every subject for this ROI
roicorr_file = 'pieman_a1_3mm_PiemanIntact_roicorr.mat';
r = load(fullfile(roicorrpath,roicorr_file));
for n = 1:length(r.roicorr), r.otherstc(:,n) = nanmean(r.roitc(:,setdiff([1:size(r.roitc,2)],n)),2); end
figure(9091); clf; set(gcf,'Color',[1 1 1]); w = 60;
for n = 1:length(r.roicorr)
    i1 = n*w-w; i2 = n*w+w;
    lagcc1 = lagcorr(r.roitc(:,n),r.otherstc(:,n),[-1*w:w]);
    plot([i1:i2],lagcc1,'b','LineWidth',2); hold on
    m1 = plot(n*w,lagcc1(w+1),'ok','LineWidth',2);
end
set(gca,'FontSize',16,'XTick',[w:w:w*n],'XTickLabel',[1:n]); grid on
xlim([0 n*w+w]); xlabel('Subject #'); ylabel('Lagged Correlation');
title(['Each Subj x Mean-of-Others']);
yr = ylim(gca); yd = (yr(2)-yr(1))/10;
plot([0 w],[-1*yd -1*yd],'k-','LineWidth',4); text(1,-2*yd,[num2str(w) ' TRs']);


%% Correlation Maps - Within Group
% input: 4d functional nifti files for subj01, subj02, subj03
% output: a single correlation map of subj01, subj02, subj03 (one-to-avg-others)
% optional outputs: mean of all subjects as nifti (4d), avg others for each subject as nifti (4d)

condname = 'PiemanIntact';
funcnames{1} = fullfile(basepath,'subjects','subj01',condname,'trans_filtered_func_data.nii');
funcnames{2} = fullfile(basepath,'subjects','subj02',condname,'trans_filtered_func_data.nii');
funcnames{3} = fullfile(basepath,'subjects','subj03',condname,'trans_filtered_func_data.nii');
funcnames{4} = fullfile(basepath,'subjects','subj04',condname,'trans_filtered_func_data.nii');
opts.outputPath = fullfile(basepath,'intersubj','corrmap');
opts.outputName = condname;
opts.crop_beginning = 10; % number of TRs to crop from beginning
opts.crop_end = 10; % number of TRs to crop from end
opts.crop_special = [1 6 14; 2 12 8; 3 11 9]; % specify different crops for different subjects, otherwise defaults to crop_beginning and crop_end
opts.mask = fullfile(basepath,'standard','MNI152_T1_3mm_brain_mask.nii'); % mask image is optional
opts.standard = fullfile(basepath,'standard','MNI152_T1_3mm_brain.nii'); % all hdr info will come from this file except for datatype=16 and bitpix=32
opts.mcutoff = 6000; % 6000 for Skyra data
opts.load_nifti = 0; % load data from nifti (specified in funcnames), save as .mat
opts.calc_avg = 1; % calculate the avg of others for each subject
opts.calc_corr = 1; % calculate intersubject correlations and save output maps as nifti
opts.save_avgothers_nii = 1; % save the avg others for each subject as nifti files (otherwise saves in .mat only)
opts.save_mean_nii = 1; % save the mean of all subjectss as a nifti file (otherwise does not calculate the mean)
nkit_nifti_corrmap(funcnames, [], opts);


%% Correlation Maps - Between Group
% input: 4d functional nifti files for subj01, subj02, subj03
% output: a single correlation map of subj01, subj02, subj03 (each-group1-indiv-to-avg-of-group2)
%  don't forget to set opts.outputName2 and opts.crop_special2
% optional outputs: mean of all subjects as nifti (4d), avg others for each subject as nifti (4d)
condname = 'PiemanIntact';
group1name = 'PieGroupA';
group2name = 'PieGroupB';
% group 1
funcnames1{1} = fullfile(basepath,'subjects','subj01',condname,'trans_filtered_func_data.nii');
funcnames1{2} = fullfile(basepath,'subjects','subj02',condname,'trans_filtered_func_data.nii');
% group 2
funcnames2{1} = fullfile(basepath,'subjects','subj03',condname,'trans_filtered_func_data.nii');
funcnames2{2} = fullfile(basepath,'subjects','subj04',condname,'trans_filtered_func_data.nii');
opts.outputPath = fullfile(basepath,'intersubj','corrmap');
opts.outputName = group1name;
opts.outputName2 = group2name;
opts.crop_beginning = 10; % number of TRs to crop from beginning
opts.crop_end = 10; % number of TRs to crop from end
opts.crop_special = [1 6 14; 2 12 8]; % specify different crops for different subjects, otherwise defaults to crop_beginning and crop_end
opts.crop_special2 = [1 11 9];
opts.mask = fullfile(basepath,'standard','MNI152_T1_3mm_brain_mask.nii'); % mask image is optional
opts.standard = fullfile(basepath,'standard','MNI152_T1_3mm_brain.nii'); % all hdr info will come from this file except for datatype=16 and bitpix=32
opts.mcutoff = 6000; % 6000 for Skyra data
opts.load_nifti = 0; % load data from nifti (specified in funcnames), save as .mat
opts.calc_avg = 1; % calculate the avg of others for each subject
opts.calc_corr = 1; % calculate intersubject correlations and save output maps as nifti
opts.save_avgothers_nii = 1; % save the avg others for each subject as nifti files (otherwise saves in .mat only)
opts.save_mean_nii = 1; % save the mean of all subjectss as a nifti file (otherwise does not calculate the mean)
nkit_nifti_corrmap(funcnames1, funcnames2, opts);


%% Regress a vector on a 4d functional
% for example, regress the audio envelope of a story onto the functional data for a subject
% saves output map in the same folder as original functional data

load(fullfile(basepath,'standard','pieman_intact_audenv.mat'));
condname = 'PiemanIntact';
intact_audenv = audenv;
funcname = fullfile(basepath,'subjects','subj01',condname,'trans_filtered_func_data.nii');
opts.outputName = 'subj01_audenv';
opts.crop_beginning = 6; % number of TRs to crop from beginning
opts.crop_end = 14; % number of TRs to crop from end
nkit_nifti_regressmap(funcname, intact_audenv, opts);


%% Crop nifti files
% output is saved in same directory as original file

crop_begin = 6;
crop_end = 14;
filename = fullfile(basepath,'subjects','subj01',condname,'trans_filtered_func_data.nii');
nkit_crop_nifti(filename,crop_begin,crop_end);


%% Calculate mean and stdev maps for a set of subjects

condname = 'PiemanIntact';
funcnames{1} = fullfile(basepath,'subjects','subj01',condname,'trans_filtered_func_data.nii');
funcnames{2} = fullfile(basepath,'subjects','subj02',condname,'trans_filtered_func_data.nii');
funcnames{3} = fullfile(basepath,'subjects','subj03',condname,'trans_filtered_func_data.nii');
funcnames{4} = fullfile(basepath,'subjects','subj04',condname,'trans_filtered_func_data.nii');
opts.crop_beginning = 10; % number of TRs to crop from beginning
opts.crop_end = 10; % number of TRs to crop from end
opts.crop_special = [1 6 14; 2 12 8; 3 11 9]; % specify different crops for different subjects, otherwise defaults to crop_beginning and crop_end
opts.standard = fullfile(basepath,'standard','MNI152_T1_3mm_brain.nii'); % hdr info will come from this file
opts.mcutoff = 6000; % 6000 for Skyra data
nkit_nifti_mean_and_std(funcnames,opts);











