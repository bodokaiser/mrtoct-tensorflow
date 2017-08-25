function write_dicom(header, volume, filepath)

meta = struct('PatientPosition', 'HFS', ...
    'BitsAllocated', 16, ...
    'BitsStored', 16, ...
    'HighBit', 15, ...
    'PixelRepresentation', 1, ...
    'SmallestPixelValue', -32768, ...
    'LargestPixelValue', +32767);
meta.Rows = header.Rows;
meta.Slices = header.Slices;
meta.Columns = header.Columns;

meta.Modality = header.Modality;
meta.PixelSpacing = header.PixelSize;
meta.SliceThickness = header.SliceThickness;

% defines dicom class
% http://dicom.nema.org/Dicom/2013/output/chtml/part04/sect_B.5.html
if meta.Modality == 'CT'
    meta.SOPClassUID = '1.2.840.10008.5.1.4.1.1.2';
end
if meta.Modality == 'MR'
    meta.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4';
end

% we assume patient is aligned with axes
meta.ImageOrientationPatient = [1 0 0 0 1 0];

vol = uint16(volume-min(volume(:)));

mkdir(filepath);

meta.FrameOfReferenceUID = dicomuid();
meta.SeriesInstanceUID = dicomuid();
meta.AcquisitionNumber = 1;

for i = 1:meta.Slices
    meta.SliceLocation = meta.SliceThickness*(i-meta.Slices/2);
    meta.InstanceNumber = i;
    meta.SOPInstanceUID = dicomuid();
    meta.ImagePositionPatient = [0 0 meta.SliceLocation];
    
    dicomwrite(vol(:, :, i), ...
        fullfile(filepath, sprintf('%06d.dcm', i)), ...
        meta, 'CreateMode', 'Copy');
end
end