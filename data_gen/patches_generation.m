function [data, labels, set] = patches_generation(scales,size_input,size_label,stride,folder,mode,training_task)

labels = [];
tmp = im2uint8(rand([size_label, size_label, 3, 80], 'single'));

for ii = 1:2
    labels = cat(4, tmp, labels);
end

data = [];


count = 0;
padding = abs(size_input - size_label)/2;
ext               =  {'*.jpg','*.png','*.bmp'};
for f_idx = 1:numel(folder)
    folder_cur = folder{f_idx};
    filepaths           =  [];
    for i = 1 : length(ext)
        filepaths = [filepaths; dir(fullfile(folder_cur, ext{i}))];
    end

    for i = 1 : length(filepaths)
        im = (imread(fullfile(folder_cur,filepaths(i).name)));
        if strcmp(training_task, 'denoising')
            im = rgb2ycbcr(im);
        else
            im = rgb2gray(im);
        end
        ss = [0.5 0.25 1];
        for s = ss
            im_label = imresize(im, s, 'bicubic');
            for j = 1:numel(scales)
                [hei,wid,~] = size(im_label);
                if hei<size_label || wid<size_label
                    continue;
                end
                [ stride1, stride2 ] = stride_init( hei, wid, 600, stride, size_label);
                for x = 1 : stride1 : (hei-size_input+1)
                    for y = 1 :stride2 : (wid-size_input+1)
                        subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1, :);
                        count=count+1;  
                        labels(:, :, :, count) = subim_label;
                    end
                end
            end
        end
        fprintf('\nImage %d patch %d\n', i, count);
    end
end
labels = labels (:,:,:,1:count);
% data = data (:,:,:,1:count);
order  = randperm(size(labels,4));
% data   = data(:, :, :, order);
labels = labels(:, :, :, order);
set    = uint8(ones(1,size(labels,4)));
if mode == 1
    set = uint8(2*ones(1,size(data,4)));
end

