clear;close all;
%% settings
folder_data = '../../dataset/rec_yuvQ22_DF';
folder_label = '../../dataset/oriyuv'
savepath = ['test_data_Y_',folder_data(end-5:end),'.h5'];

size_input = 35;
size_label = 35;
stride = 35;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
padding = abs(size_input - size_label)/2;
test_num = 100;
count = 0;

%% generate data
filepaths = dir(fullfile(folder_data,'*.yuv'));

fid_set1 = fopen('../../dataset/set1.txt');
fid_set2 = fopen('../../dataset/set2.txt');
size_yuv = zeros(1,2,length(filepaths));
while feof(fid_set1) ~=1
    line = fgetl(fid_set1);
    index = str2num(line(1:5));
    size_yuv(:,:,index) = [512,384]; 
end
fclose(fid_set1);
while feof(fid_set2) ~=1
    line = fgetl(fid_set2);
    index = str2num(line(1:5));
    size_yuv(:,:,index) = [384,512]; 
end
fclose(fid_set2);

   % for i = 1 : length(filepaths)-test_num
    for i = length(filepaths)-test_num+1 : length(filepaths)
        fprintf('yuv: %d\n',i);
        fid = fopen(fullfile(folder_data,filepaths(i).name),'rb');
        ImgData_Y = fread(fid, size_yuv(:,:,i), 'uint8');      % image data Y
        ImgData_U = fread(fid ,size_yuv(:,:,i)/2, 'uint8');  % image data U
        ImgData_V = fread(fid ,size_yuv(:,:,i)/2, 'uint8');  % image data V
        ImgData_Y = im2double(uint8(ImgData_Y'));
        ImgData_U = im2double(uint8(ImgData_U'));
        ImgData_V = im2double(uint8(ImgData_V'));
        fid_label = fopen(fullfile(folder_label,filepaths(i).name),'rb');
        Imglabel_Y = fread(fid_label, size_yuv(:,:,i), 'uint8');      % image data Y
        Imglabel_U = fread(fid_label ,size_yuv(:,:,i)/2, 'uint8');  % image data U
        Imglabel_V = fread(fid_label ,size_yuv(:,:,i)/2, 'uint8');  % image data V
        Imglabel_Y = im2double(uint8(Imglabel_Y'));
        Imglabel_U = im2double(uint8(Imglabel_U'));
        Imglabel_V = im2double(uint8(Imglabel_V'));
        [hei,wid] = size(ImgData_Y);
        for x = 1 : stride : hei-size_input+1
            for y = 1 :stride : wid-size_input+1

                subim_input = ImgData_Y(x : x+size_input-1, y : y+size_input-1);
                subim_label = Imglabel_Y(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);

                count=count+1;
                data(:, :, 1, count) = subim_input;
                label(:, :, 1, count) = subim_label;
            end
        end
        fclose(fid);
        fclose(fid_label);
    end
%end

% order = randperm(count);
% data = data(:, :, 1, order);
% label = label(:, :, 1, order);

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz);
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);