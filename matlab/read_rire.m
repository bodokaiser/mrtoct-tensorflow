function [header, volume] = read_rire(headerpath, volumepath)
header = read_rire_header(headerpath);
volume = read_rire_volume(volumepath, header);
end