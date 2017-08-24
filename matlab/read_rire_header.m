function outStruct = read_rire_header(filename)
% Copyright 2012 The MathWorks, Inc.
fid = fopen(filename);

thisLine = fgetl(fid);

outStruct = struct();

while ischar(thisLine)    
    [startIndex, endIndex] = regexp(thisLine, ' := ');
    
    switch (lower(thisLine(1:(startIndex-1))))
        case 'modality'
            outStruct.Modality = thisLine((endIndex+1):end);
        case 'slice thickness'
            outStruct.SliceThickness = str2double(thisLine(endIndex:end));
        case 'pixel size'
            [startIdx,endIdx] = regexp(thisLine,' : ');
            outStruct.PixelSize(1) = str2double(thisLine(endIndex:startIdx));
            outStruct.PixelSize(2) = str2double(thisLine((endIdx+1):end));
        case 'rows'
            outStruct.Rows = str2double(thisLine(endIndex:end));
        case 'columns'
            outStruct.Columns = str2double(thisLine(endIndex:end));
        case 'slices'
            outStruct.Slices  = str2double(thisLine(endIndex:end));
    end
    
    thisLine = fgetl(fid);
    
end

fclose(fid);

end