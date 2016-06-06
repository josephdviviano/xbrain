
function nifti_roi_timecourse(datafiles, roifiles, opts)

% if roifiles is a cell array of strings, then it should be the same length
% as datafiles, and roifile{N} will be extracted from datafile{N}.
% if roifiles is a string (a single filename), that same roi will be used
% for all datafiles.
%
% opts.outputPath: location where you want to save the summary files
%
% all voxels in the roi are averaged before calculating cross-subject correlations
%

if ~isfield(opts,'mcutoff'),                opts.mcutoff = []; end
if isempty(opts.mcutoff), mcutoff = 6000; else mcutoff = opts.mcutoff; end % mean (over time) value at a voxel must be at least this high to be retained

%mcutoff = 6000; % mean (over time) value at a voxel must be at least this high to be retained
scutoff = 0.7; % at least 70% of subjects must contribute data to each voxel

if ~exist(opts.outputPath,'dir')
    [pathstr,dirname] = fileparts(opts.outputPath);
    mkdir(pathstr,dirname);
end
origpath = pwd;
cd(opts.outputPath)

if ~isfield(opts,'roidata'), opts.roidata = ''; end
if ~isfield(opts,'crop_special'), opts.crop_special = 0; end
if isempty(opts.crop_special), opts.crop_special = 0; end
if ~iscell(datafiles), datafiles = {datafiles}; end % if only one datafile (a string) is entered, reformat to cell
     
singleroi = 0;
if ~iscell(roifiles) % use the same roi for all subjects. load it now
    singleroi = 1;
    roifile = roifiles; roinii = load_nii(roifile); rdatasize = size(roinii.img);
    roi = single(roinii.img>0); % binarize roi
    roimask = single(reshape(roi,[(size(roi,1)*size(roi,2)*size(roi,3)),size(roi,4)]));
    [rpathstr rname] = fileparts(roifile);
end
       
if strcmp(opts.roidata,'savedata')
    for sid = 1:length(datafiles)
        if ~singleroi % load the roi now
            if ~exist(roifiles{sid})
                fprintf('ROI not found -- skipping this subject.\n');
                continue
            end
            roinii = load_nii(roifiles{sid});
            rdatasize = size(roinii.img);
            roi = single(roinii.img>0); % binarize roi
            roimask = single(reshape(roi,[(size(roi,1)*size(roi,2)*size(roi,3)),size(roi,4)]));
            [rpathstr rname] = fileparts(roifiles{sid});
        end
        
        if ~exist(datafiles{sid}) && ~exist([datafiles{sid}(1:end-3),'mat'])
            fprintf('Functional data not found -- skipping this subject.\n');
            continue
        end
        
        % data can be either .nii or .mat
        % if mat version exists, load it
        if exist([datafiles{sid}(1:end-3),'mat'])
            fprintf(['Loading subj ' num2str(sid) ' mat file\n']);
            load([datafiles{sid}(1:end-3) 'mat']);
        else % otherwise load the nifti and save a .mat version for next time
            fprintf(['Loading subj ' num2str(sid) ' Nifti file\n']);
            nii = load_nii(datafiles{sid});
            data = nii.img;
            datasize = size(data);
            nii.img = [];
            data = single(reshape(data,[(size(data,1)*size(data,2)*size(data,3)),size(data,4)]));
            mdata = mean(data,2);
            keptvox = mdata>mcutoff;
            save([datafiles{sid}(1:end-3) 'mat'],'data','datasize','keptvox');
        end

        if (datasize(1)~=rdatasize(1))||(datasize(2)~=rdatasize(2))||(datasize(3)~=rdatasize(3))
            fprintf('ROI dimensions do not match functional data -- skipping this subject.\n');
            continue
        end
        
        rdata = data(logical(roimask),:);
        mdata = mean(rdata,2);
        keptvox = mdata>mcutoff;
        
        % save the roidata uncropped
        [dpathstr dname] = fileparts(datafiles{sid});
        save(fullfile(dpathstr,[rname '_' dname '.mat']),'rdata','keptvox');
    end
    return % if using savedata flag, stop here
end

% if using loaddata flag, start here
nansubs = [];
for sid = 1:length(datafiles)
    if ~singleroi, [rpathstr rname] = fileparts(roifiles{sid}); end
    [dpathstr dname] = fileparts(datafiles{sid});
    fname = fullfile(dpathstr,[rname '_' dname '.mat']);
    if ~exist(fname)
        fprintf(['File not found -- skipping this subject.\n']);
        nansubs = [nansubs sid];
    else
        load(fullfile(dpathstr,[rname '_' dname '.mat']));
        all_rdata{sid} = rdata;
        all_keptvox{sid} = keptvox;
    end
end
roitc = []; uncropped_roitc = [];
for sid = 1:length(datafiles)
    if ismember(sid,nansubs), continue; end
    rdata = all_rdata{sid};
    rdata2 = crop_roitc(sid,rdata,opts);
    % apply 6000-value cutoff but not 70%-subject cutoff
    subj_roitc = nanmean(zscore(rdata2(logical(all_keptvox{sid}),:)')');
    roitc(:,sid) = zscore(subj_roitc);
    
    % also save an uncropped timecourse (but do shift it)
    nopts.crop_beginning = 0;
    nopts.crop_end = 0;
    nopts.crop_special = opts.crop_special;
    if numel(nopts.crop_special)>1,
        nopts.crop_special(:,2) = nopts.crop_special(:,2) - opts.crop_beginning;
        nopts.crop_special(:,3) = nopts.crop_special(:,3) - opts.crop_end;
    end
    rdata3 = crop_roitc(sid,rdata,nopts);
    nsubj_roitc = nanmean(zscore(rdata3(logical(all_keptvox{sid}),:)')');
    uncropped_roitc(:,sid) = zscore(nsubj_roitc);
end    
roitc(:,nansubs) = NaN;
meantc = nanmean(roitc,2);

for sid = 1:length(datafiles)
    if ismember(sid,nansubs), continue; end
    others = setdiff([1:size(roitc,2)],sid);
    roicorr(sid) = corr(roitc(:,sid),nanmean(roitc(:,others),2));
end
roicorr(nansubs) = NaN;
meancorr = nanmean(roicorr);
if ~isfield(opts,'subgroup'), opts.subgroup = ''; end
if ~isempty(opts.subgroup), opts.subgroup = ['_' opts.subgroup]; end
savename = [rname '_' opts.outputName opts.subgroup '_roicorr.mat'];
save(savename,'roicorr','roitc','meancorr','meantc','uncropped_roitc');

cd(origpath);
end

function subj_roitc = crop_roitc(sid,subj_roitc,opts)
if ~ismember(sid,opts.crop_special(:,1))
    subj_roitc(:,1:opts.crop_beginning) = []; % crop TRs from beginning of time series
    subj_roitc(:,end-opts.crop_end+1:end) = []; % crop TRs from end of time series
elseif ismember(sid,opts.crop_special(:,1)) % crop different begin/end TRs for specified subjects
    cid = find(sid==opts.crop_special(:,1));
    crop_begin = opts.crop_special(cid,2);
    crop_end = opts.crop_special(cid,3);
    if crop_begin<0 % adds timepoints to the beginning of series if needed
        for k = 1:crop_begin*-1
            subj_roitc(:,2:end+1) = subj_roitc;
        end
    end
    subj_roitc(:,1:crop_begin) = []; % crop TRs from beginning of time series
    if crop_end<0
        for k = 1:crop_end*-1
            subj_roitc(:,end+1) = subj_roitc(:,end);
        end
    end
    subj_roitc(:,end-crop_end+1:end) = []; % crop TRs from end of time series
end
end






















