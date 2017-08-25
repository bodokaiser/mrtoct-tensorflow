% path to data directory
datadir = '../data';

% volume directories
ctdir = 'ct';
mrdir = 'mr_T1';

filenames = dir(datadir);
filenames(~[filenames.isdir]) = [];

header_filename = 'header.ascii';
volume_filename = 'image.bin';
niidef_filename = 'sanon-0001-00001-000001.nii';

olddir = pwd();

addpath('spm12');
addpath('.');

for i = 1:numel(filenames)
    filename = filenames(i);
    filepath = fullfile(filename.folder, filename.name);
    
    ctheader_path = fullfile(filepath, ctdir, header_filename);
    ctvolume_path = fullfile(filepath, ctdir, volume_filename);
    
    mrheader_path = fullfile(filepath, mrdir, header_filename);
    mrvolume_path = fullfile(filepath, mrdir, volume_filename);
    
    ctworkpath = tempname;
    mrworkpath = tempname;
    
    % not every patient has CT and MRI volumes
    if exist(ctheader_path, 'file') == 2 && exist(ctvolume_path, 'file') == 2
        if exist(mrheader_path, 'file') == 2 && exist(mrvolume_path, 'file') == 2
            % convert rire to dicom
            [ctheader, ctvolume] = read_rire(ctheader_path, ctvolume_path);
            [mrheader, mrvolume] = read_rire(mrheader_path, mrvolume_path);
            write_dicom(ctheader, ctvolume, ctworkpath);
            write_dicom(mrheader, mrvolume, mrworkpath);
            
            % convert ct dicom to nii
            cd(ctworkpath);
            hdrref = spm_dicom_headers(list_filepaths('*.dcm'));
            spm_dicom_convert(hdrref);
            movefile(niidef_filename, fullfile(tempdir, 'ct.nii'));
            rmdir(ctworkpath, 's');
            
            % convert mr dicom to nii
            cd(mrworkpath);
            hdrmov = spm_dicom_headers(list_filepaths('*.dcm'));
            spm_dicom_convert(hdrmov);
            movefile(niidef_filename, fullfile(tempdir, 'mr.nii'));
            rmdir(mrworkpath, 's');
            
            % coregister dicoms
            cd(tempdir);
            ref = 'ct.nii';
            mov = 'mr.nii';
            refvol = spm_vol(ref);
            movvol = spm_vol(mov);
            x = spm_coreg(refvol, movvol);
            M = spm_matrix(x);
            MM = zeros(4,4,1);
            MM(:,:,1) = spm_get_space(ref);
            spm_get_space(ref, M*MM(:,:,1));
            spm_reslice({ref; mov}, struct('interp', 4, 'mean', 0, 'which', 1));
            spm_file(mov, 'prefix', 'r');
            
            % move final files back to data
            movefile('ct.nii', fullfile(filepath, 'ct.nii'));
            movefile('rmr.nii', fullfile(filepath, 'mr.nii'));
        end
    end
end

cd(olddir);