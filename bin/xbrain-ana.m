clear all; clc

%% OPTIONS
conditions = {'im', 'ob', 'rest'};
n_rois = 268;
%subjlist = '/projects/jdv/data/imob/working/lists/all-sz.csv';
subjlist = '/projects/jdv/data/imob/working/lists/all-hc.csv';

%% INIT
% add the ISC toolbox to our path
addpath(genpath('/projects/jdv/code/xbrain/bin/isc_nifti_kit'));

% define our output paths
datapath = '/projects/jdv/data/imob/working/data';
assetpath = '/projects/jdv/data/imob/working/assets';
roidataflag = 'savedata'; % 'loaddata': run on .mat (faster), 'savedata': run on .nii

%% ROI-wise intersubject correlation
for c = conditions;

    c = char(c); % convert c to a string

    % Load data
    fid = fopen(subjlist); % import subjects
    data = fread(fid, '*char')';
    fclose(fid);
    subjects = regexp(data, '\n', 'split');

    if isempty(subjects{end}) == 1;
        subjects = subjects(1:end-1); % strip off empty final line if required.
    end

    n_subj = length(subjects);

    for n = 1:n_rois;
        rois{n} = [assetpath '/anat_MNI_shen_268-resamp-' int2str(n) '.nii']; % load ROI niftis
    end

    if strcmp(c, 'im') == 1 | strcmp(c, 'ob') == 1;
        for n = 1:n_subj;
        	funcnames{n} = [datapath '/' subjects{n} '_' c '_glm_residuals.nii'];
        end

    elseif strcmp(c, 'rest') == 1;
        for n = 1:n_subj;
            funcnames{n} = [datapath '/' subjects{n} '_' c '.nii'];
        end
    end

    % Analysis: outputs 'roi_condname_roicorr.mat' that contains ISC info for the ROI
    for rr = 1:length(rois)
        roi = rois{rr};
        roinames = roi;
        fprintf(['cond = ' c ', roi = ' num2str(rr) '\n']);

        opts.outputPath = fullfile(datapath, 'roicorr'); % create this folder in advance
        opts.outputName = c;
        opts.crop_beginning = 0; % number of TRs to crop from beginning
        opts.crop_end = 0; % number of TRs to crop from end
        opts.mcutoff = 100; % mean over time to be retained (default is far too high for our centre).
        opts.roidata = roidataflag;

        nkit_nifti_roi_timecourse(funcnames, roinames, opts);

        if strcmp(roidataflag, 'savedata') % if using savedata flag, auto-run loaddata afterward
            opts.roidata = 'loaddata';
            nkit_nifti_roi_timecourse(funcnames, roinames, opts);
        end
    end
end

% %% Correlation Maps - Within Group
% % input: 4d functional nifti files for subj01, subj02, subj03
% % output: a single correlation map of subj01, subj02, subj03 (one-to-avg-others)
% % optional outputs: mean of all subjects as nifti (4d), avg others for each subject as nifti (4d)

% opts.outputPath = fullfile(datapath,'intersubj','corrmap');
% opts.outputName = c;
% opts.crop_beginning = 0; % number of TRs to crop from beginning
% opts.crop_end = 0; % number of TRs to crop from end
% %opts.crop_special = [1 6 14; 2 12 8; 3 11 9]; % specify different crops for different subjects, otherwise defaults to crop_beginning and crop_end
% %opts.mask = fullfile(basepath,'standard','MNI152_T1_3mm_brain_mask.nii'); % mask image is optional
% opts.standard = fullfile(basepath,'standard','MNI152_T1_3mm_brain.nii'); % all hdr info will come from this file except for datatype=16 and bitpix=32
% opts.mcutoff = 6000; % 6000 for Skyra data
% opts.load_nifti = 0; % load data from nifti (specified in funcnames), save as .mat
% opts.calc_avg = 1; % calculate the avg of others for each subject
% opts.calc_corr = 1; % calculate intersubject correlations and save output maps as nifti
% opts.save_avgothers_nii = 1; % save the avg others for each subject as nifti files (otherwise saves in .mat only)
% opts.save_mean_nii = 1; % save the mean of all subjectss as a nifti file (otherwise does not calculate the mean)
% nkit_nifti_corrmap(funcnames, [], opts);

%% DATA EXPORT
x = load([datapath '/roicorr/hc/anat_MNI_shen_268-resamp-1_im_roicorr.mat']);
n_hc = length(x.roicorr);

x = load([datapath '/roicorr/sz/anat_MNI_shen_268-resamp-1_im_roicorr.mat']);
n_sz = length(x.roicorr);

hc_corr_im = zeros(n_rois, n_hc);
hc_corr_ob = zeros(n_rois, n_hc);
hc_corr_rest = zeros(n_rois, n_hc);

sc_corr_im = zeros(n_rois, n_sz);
sc_corr_ob = zeros(n_rois, n_sz);
sc_corr_rest = zeros(n_rois, n_sz);

groups = {'hc', 'sz'};
count = 1;

for group = groups;
    group = char(group);
    datadir = [datapath '/roicorr/' group];

    for roi = 1:n_rois;

        % imitate
        x = load([datadir '/anat_MNI_shen_268-resamp-' int2str(roi) '_im_roicorr.mat']);
        if strcmp(group, 'hc') == 1;
            hc_corr_im(roi, :) = x.roicorr;
        elseif strcmp(group, 'sz') == 1;
            sc_corr_im(roi, :) = x.roicorr;
        end

        % observe
        x = load([datadir '/anat_MNI_shen_268-resamp-' int2str(roi) '_ob_roicorr.mat']);
        if strcmp(group, 'hc') == 1;
            hc_corr_ob(roi, :) = x.roicorr;
        elseif strcmp(group, 'sz') == 1;
            sc_corr_ob(roi, :) = x.roicorr;
        end

        % rest
        x = load([datadir '/anat_MNI_shen_268-resamp-' int2str(roi) '_rest_roicorr.mat']);
        if strcmp(group, 'hc') == 1;
            hc_corr_rest(roi, :) = x.roicorr;
        elseif strcmp(group, 'sz') == 1;
            sc_corr_rest(roi, :) = x.roicorr;
        end
        disp(roi)
    end
end

% format a matrix for export (remove subjects with missing values)
hc_data = zeros(n_rois,1);
for subj = 1:length(hc_corr_im(1,:));
    if sum(isnan(hc_corr_im(:,subj))) ~= n_rois ;
        hc_data = [hc_data, hc_corr_im(:,subj)];
        hc_data = [hc_data, hc_corr_ob(:,subj)];
        hc_data = [hc_data, hc_corr_rest(:,subj)];
    end
end
hc_data = hc_data(:, 2:end);

sc_data = zeros(n_rois,1);
for subj = 1:length(sc_corr_im(1,:));
    if sum(isnan(sc_corr_im(:,subj))) ~= n_rois;
        sc_data = [sc_data, sc_corr_im(:,subj)];
        sc_data = [sc_data, sc_corr_ob(:,subj)];
        sc_data = [sc_data, sc_corr_rest(:,subj)];
    end
end
sc_data = sc_data(:, 2:end);

% id, group, condition, roi, correlation
output = zeros(1,5);

n_hc = length(hc_data(1,:))/3;
n_sc = length(sc_data(1,:))/3;
n_rois = length(hc_data(:,1));

count = 0;
for subj = 1:n_hc;
    for roi = 1:n_rois;
        for condition = [1,2,3];
            data = zeros(1,5);
            data(1) = count + 1; % id
            data(2) = 1; % group
            data(3) = condition; % 1 = im, 2 = ob, 3 = rest
            data(4) = roi; % roi
            data(5) = hc_data(roi, 3*count+condition); % correlation
            output = [output; data]; % append data to output matrix
        end
    end
    disp([int2str(3*count+1) '-' int2str(3*count+3) ' / ' int2str(n_hc*3)])
    count = count+1;
end

for subj = 1:n_sc;
    for roi = 1:n_rois;
        for condition = [1,2,3];
            data = zeros(1,5);
            data(1) = count + 1; % id
            data(2) = 2; % group
            data(3) = condition; % 1 = im, 2 = ob, 3 = rest
            data(4) = roi; % roi
            data(5) = sc_data(roi, 3*(count-n_hc)+condition); % correlation
            output = [output; data]; % append data to output matrix
        end
    end
    disp([int2str(3*(count-n_hc)+1) '-' int2str(3*(count-n_hc)+3) ' / ' int2str(n_sc*3)])
    count = count+1;
end

output = output(2:end, :);

% write out the data
txt = sprintf('id,group,condition,roi,correlation');
dlmwrite('xcor-data.csv', txt, '');
dlmwrite('xcor-data.csv', output, '-append', 'delimiter', ',', 'precision', 20);
