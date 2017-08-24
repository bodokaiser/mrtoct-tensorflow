function outStruct = read_rire(filepath)
header = read_rire_header(fullfile(filepath, 'header.ascii'));
volume = read_rire_volume(fullfile(filepath, 'image.bin'), header);
outStruct = struct('Header', header, 'Volume', volume);
end