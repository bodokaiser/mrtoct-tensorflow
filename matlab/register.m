refs = spm_select([1, Inf], 'image', 'select ct nifti images', ...
    {}, '../data/nifti', 'ct.*nii');
movs = spm_select([1, Inf], 'image', 'select mr nifti images', ...
    {}, '../data/nifti', 'mr.*nii');

refs = sort(cellstr(refs));
movs = sort(cellstr(movs));

if numel(refs) ~= numel(movs)
    error('inconsistent number of volumes')
end

for i = 1:numel(refs)
    ref = char(refs(i));
    mov = char(movs(i));

    x = spm_coreg(ref, mov);

    M = spm_matrix(x);
    MM = zeros(4,4);
    MM(:,:) = spm_get_space(mov);
    spm_get_space(mov,M\MM(:,:));
    spm_reslice({ref; mov}, struct('interp',4,'mean',0,'which',1,...
        'prefix','co-'));
end