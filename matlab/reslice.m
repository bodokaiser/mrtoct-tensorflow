addpath('spm12');

refs = spm_select([1, Inf], 'image', 'select ct nifti images', ...
    {}, '../data/nifti', 'ct.*nii');
movs = spm_select([1, Inf], 'image', 'select mr nifti images', ...
    {}, '../data/nifti', 'mr.*nii');

refs = unique(sort(cellstr(refs)));
movs = unique(sort(cellstr(movs)));

if numel(refs) ~= numel(movs)
    error('inconsistent number of volumes')
end

voxsiz = [.65 .65 .65];

refvols = spm_vol(char(refs));
movvols = spm_vol(char(movs));

for i=1:numel(movvols)
    refvol = refvols(i);
    movvol = movvols(i);
    
    bb = spm_get_bbox(movvol, 1000);
   
   VV1(1:2) = refvol;
   VV2(1:2) = movvol;
   
   VV1(1).mat = spm_matrix([bb(1,:) 0 0 0 voxsiz])*spm_matrix([-1 -1 -1]);
   VV1(1).dim = ceil(VV1(1).mat \ [bb(2,:) 1]' - 0.1)';
   VV1(1).dim = VV1(1).dim(1:3);
   VV2(1).mat = VV1(1).mat;
   VV2(1).dim = VV1(1).dim;
   
   spm_reslice(VV1,struct('mean',false,'which',1,'interp',4,'prefix','re-'));
   
   %VV2(1).mat = spm_matrix([bb(1,:) 0 0 0 voxsiz])*spm_matrix([-1 -1 -1]);
   %VV2(1).dim = ceil(VV2(1).mat \ [bb(2,:) 1]' - 0.1)';
   %VV2(1).dim = VV2(1).dim(1:3);
   spm_reslice(VV2,struct('mean',false,'which',1,'interp',4,'prefix','re-'));
end