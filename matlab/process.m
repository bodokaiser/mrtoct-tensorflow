filenames = dir('data');
filenames(~[filenames.isdir]) = [];

ctDirectory = 'ct';
mrDirectory = 'mr_T1';

headerFilename = 'header.ascii';
volumeFilename = 'image.bin';

paths = [];

for i = 1:numel(filenames)
    filename = filenames(i);
    filepath = fullfile(filename.folder, filename.name);
    
    ctPath = fullfile(filepath, ctDirectory);
    ctHeaderPath = fullfile(ctPath, headerFilename);
    ctVolumePath = fullfile(ctPath, volumeFilename);
    
    mrPath = fullfile(filepath, mrDirectory);
    mrHeaderPath = fullfile(mrPath, headerFilename);
    mrVolumePath = fullfile(mrPath, volumeFilename);
    
    if exist(ctHeaderPath, 'file') == 2 && exist(ctVolumePath, 'file') == 2
        if exist(mrHeaderPath, 'file') == 2 && exist(mrVolumePath, 'file') == 2
            paths = [paths struct('ctPath', ctPath, 'mrPath', mrPath)];
        end
    end
end

paths