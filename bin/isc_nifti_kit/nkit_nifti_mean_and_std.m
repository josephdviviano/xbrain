
function nifti_mean_and_std(datafiles,opts)

% given a list of 4D data files, outputs a mean volume (3D) and a volume of std dev values.
% output is in same directory as original file
%
% opts MUST contain the fields:
%  crop_beginning: how much to crop from beginning
%  crop_end: how much to crop from end
%  crop_special: % specify different crops for different subjects
%                 [<subjnum> <crop_begin> <crop_end>; <subjnum> <crop_begin> <crop_end>; etc.] 
%                 otherwise defaults to crop_beginning and crop_end
%  standard: the standard brain you normalized to, probably MNI152_T1_3mm_brain.nii
%
% j chen 07/01/13

if ~isfield(opts,'mcutoff'),                opts.mcutoff = []; end
if isempty(opts.mcutoff), mcutoff = 6000; else mcutoff = opts.mcutoff; end % mean (over time) value at a voxel must be at least this high to be retained
if ~isfield(opts,'crop_special'),           opts.crop_special = 0; end
if isempty(opts.crop_special),              opts.crop_special = 0; end
if ~iscell(datafiles),                      datafiles = {datafiles}; end

origdir = pwd;

for sid = 1:length(datafiles)
    if ~exist(datafiles{sid})
        fprintf('Functional data not found -- skipping this subject.\n');
        continue
    end
    pathstr = fileparts(datafiles{sid});
    % data can be either .nii or .mat. if mat version exists, load it
    if exist([datafiles{sid}(1:end-3),'mat'])
        fprintf(['Loading subj ' num2str(sid) ' mat file\n']);
        load([datafiles{sid}(1:end-3) 'mat']);
    else % otherwise load the nifti and save a .mat version for next time
        fprintf(['Loading subj ' num2str(sid) ' Nifti file\n']);
        nii = load_nii(datafiles{sid});
        data = nii.img; datasize = size(data); nii.img = [];
        data = single(reshape(data,[(size(data,1)*size(data,2)*size(data,3)),size(data,4)]));
        mdata = mean(data,2);
        keptvox = mdata>mcutoff;
        filename = [opts.outputName '_subj_' num2str(sid)]
        save([datafiles{sid}(1:end-3) 'mat'],'data','datasize','keptvox');
    end
    
    % data are saved as voxelsXtimepoints 2D, so we reshape to 3D to crop
    data = reshape(data,[datasize(1), datasize(2), datasize(3), datasize(4)]);
    data = crop_data(sid,data,opts);
    data = reshape(data,[(size(data,1)*size(data,2)*size(data,3)),size(data,4)]);
    data(~keptvox) = NaN;
        
    stdvdata = std(data'); % variance per voxel
    stdvdata2 = reshape(stdvdata,datasize(1),datasize(2),datasize(3));
    meandata = mean(data'); % mean per voxel
    meandata2 = reshape(meandata,datasize(1),datasize(2),datasize(3));
    
    % save both nifti and mat versions
    nii = load_nii(opts.standard);
    cd(pathstr)
    [pathstr,fname] = fileparts(datafiles{sid});
    nii.hdr.dime.glmax = max(max(max(nii.img))); % this is not critical
    nii.hdr.dime.dim(1) = 3; % this is critical: change from 4d to 3d
    nii.hdr.dime.dim(5) = 1; % this is critical: change from 4d to 3d
    
    nii.img = stdvdata2;
    nii.hdr.dime.cal_min=0; % set the default display threshold
    nii.hdr.dime.cal_max=200;
    savename = [fname '_stdvmap.nii'];
    save_nii(nii,fullfile(pathstr,[savename '.nii']));
    
    nii.img = meandata2;
    nii.hdr.dime.cal_min=6000; % set the default display threshold
    nii.hdr.dime.cal_max=14000;
    savename = [fname '_meanmap.nii'];
    save_nii(nii,fullfile(pathstr,[savename '.nii']));
    
    save(fullfile(pathstr,[fname '_mean_and_std.mat']),'stdvdata','meandata','datasize');
    
end
cd(origdir)

end




