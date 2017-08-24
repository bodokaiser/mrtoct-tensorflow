function filepaths = listFilePaths(filepath)
filenames = dir(filepath);
filepaths = cell(1, numel(filenames));

for i = 1:numel(filenames)
    filepaths{i} = fullfile(filenames(i).folder, filenames(i).name);
end

filepaths = char(filepaths);
end

