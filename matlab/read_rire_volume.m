function volume = read_rire_volume(filename, header)
volume = multibandread(filename, [header.Rows, header.Columns, ...
    header.Slices], 'int16', 0, 'bsq', 'ieee-be');
end

