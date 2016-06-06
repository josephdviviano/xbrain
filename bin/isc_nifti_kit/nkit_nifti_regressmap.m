
function nifti_regressmap(funcname, vec, opts)

% funcname = a string indicating full path to the 4D NIFTI file
%   'Z:\janice\PieMan\piesky\subjects\JC_012312\analysis\preproc\preproc01.feat\rfiltered_func_data.nii'
%    
% opts.crop_beginning = 10; % number of TRs to crop from beginnning
% opts.crop_end = 10; % number of TRs to crop from end
% opts.outputName = 'filename'; % saves in the same dir as funcname

origpath = pwd;
[pathstr,fname] = fileparts(funcname);
cd(pathstr)

if ~isfield(opts,'mcutoff'),                opts.mcutoff = []; end
if isempty(opts.mcutoff), mcutoff = 6000; end % mean (over time) value at a voxel must be at least this high to be retained

nii = load_nii(funcname);
data = nii.img;
nii.img = [];

data(:,:,:,1:opts.crop_beginning) = []; % crop TRs from beginning of time series
data(:,:,:,end-opts.crop_end+1:end) = []; % crop TRs from end of time series

% data = single(data.*repmat(mask,[1 1 1 size(data,4)]));
datasize = size(data);
data = single(reshape(data,[(size(data,1)*size(data,2)*size(data,3)),size(data,4)]));
mdata = mean(data,2);
rejvoxels = mdata<mcutoff;
data(rejvoxels,:) = NaN;

% regress
rmap = corr(data',vec);

% reshape back to voxel space
rmap_img = reshape(rmap,[datasize(1), datasize(2), datasize(3)]);

% Save a Nifti file for the mean corr map
fprintf('Saving maps\n');
nii.img = rmap_img;
nii.img(isnan(nii.img)) = 0;
savename = opts.outputName;
if exist(savename,'file')
    system(['mv ' savename ' ' savename datestr(now, 'YYYYmmDD_HHMM')]);
end
nii.hdr.dime.glmax = max(max(max(nii.img))); % this is not critical
nii.hdr.dime.dim(1) = 3; % this is critical: change from 4d to 3d
nii.hdr.dime.dim(5) = 1; % this is critical: change from 4d to 3d
save_nii(nii,[savename '.nii']);
cd(origpath);

% keyboard


















