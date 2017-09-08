HEADER_FN = 'header.ascii';
VOLUME_FN = 'image.bin';

raw_dir = spm_input('dir to read RAW from:','1','s','../data/raw');
dcm_dir = spm_input('dir to write DICOM to:','2','s','../data/dicom');

if exist(raw_dir, 'dir') == 0
    error('raw dir (%s) does not exist', raw_dir)
end
if exist(dcm_dir, 'dir') == 0
    mkdir(dcm_dir)
end

% subdirs of raw_dir are patient dirs
pdirs = dir(raw_dir);
pdirs(~[pdirs.isdir]) = [];
pdirs(1:2) = [];

for i = 1:numel(pdirs)
    pdir = fullfile(pdirs(i).folder, pdirs(i).name);
    
    ct_dir = fullfile(pdir, 'ct');
    mr_dir = fullfile(pdir, 'mr_T1');
    
    ctheader = fullfile(ct_dir, HEADER_FN);
    mrheader = fullfile(mr_dir, HEADER_FN);
    ctvolume = fullfile(ct_dir, VOLUME_FN);
    mrvolume = fullfile(mr_dir, VOLUME_FN);
    
    if exist(ctheader, 'file') == 2 && exist(ctvolume, 'file') == 2
        if exist(mrheader, 'file') == 2 && exist(mrvolume, 'file') == 2
            [ctheader, ctvolume] = read_rire(ctheader, ctvolume);
            [mrheader, mrvolume] = read_rire(mrheader, mrvolume);
            
            ct_out_dir = fullfile(dcm_dir, sprintf('%s-%s', ...
                'ct', pdirs(i).name));
            mr_out_dir = fullfile(dcm_dir, sprintf('%s-%s', ...
                'mr', pdirs(i).name));
            
            write_dicom(ctheader, ctvolume, ct_out_dir);
            write_dicom(mrheader, mrvolume, mr_out_dir);
        end
    end
end