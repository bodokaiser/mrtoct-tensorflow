dcm_dir = spm_input('dir to read DICOM from:','1','s','../data/dicom');
nii_dir = spm_input('dir to write NIfTI to:','2','s','../data/nifti');

pdirs = dir(dcm_dir);
pdirs(~[pdirs.isdir]) = [];
pdirs(1:2) = [];

if exist(dcm_dir, 'dir') == 0
    error('dcm dir (%s) does not exist', dcm_dir)
end
if exist(nii_dir, 'dir') == 0
    mkdir(nii_dir)
end

for i = 1:numel(pdirs)    
    source = dir(fullfile(pdirs(i).folder, pdirs(i).name,'*.dcm'));
    
    target = spm_dicom_convert(spm_dicom_headers(char(...
        fullfile({source.folder}, {source.name}))));
    
     movefile(char(target.files(1)), fullfile(nii_dir, ...
         sprintf('%s.nii', pdirs(i).name)));
end
