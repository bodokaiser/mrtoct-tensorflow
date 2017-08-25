sourcedir = '../data/nii';
targetdir = '../data/png';

sourcedir = fullfile(pwd, sourcedir);
targetdir = fullfile(pwd, targetdir);

mkdir(targetdir);

filenames = dir(sourcedir);
filenames(~[filenames.isdir]) = [];
filenames(1:2) = [];

for i = 1:numel(filenames)
    filename = filenames(i);
    filepath = fullfile(filename.folder, filename.name);
    
    
    ct = spm_read_vols(spm_vol(fullfile(filepath, 'ct.nii')));
    mr = spm_read_vols(spm_vol(fullfile(filepath, 'mr.nii')));
    cb = cat(2, ct, mr);

    minValue = min(cb(:));
    maxValue = max(cb(:));

    cb = (cb - minValue) / (maxValue - minValue);

    outdir = fullfile(targetdir, filename.name);
    
    mkdir(outdir);
    
    for i = 1:size(cb, 3)
        imwrite(cb(:,:,i), fullfile(outdir, sprintf('%04d.png', i)));
    end
end

clear;