clear;close all;
%% settings
count = 0;
size_input = 35;
size_label = 35;
stride = 35;
qp = zeros(size_input, size_input, 1, 14000);
 data1 = zeros(size_input, size_input, 1, 14000);%note : re-caculate the last value accroding to your dataset size
 data2 = zeros(size_input, size_input, 2, 14000);
 label = zeros(size_label, size_label, 2, 14000);
 padding = abs(size_input - size_label)/2;
 test_num = 100;

for QP = [22 27 32 37]
    folder_data = ['../../dataset/rec_yuvQ',num2str(QP)];
    folder_label = '../../dataset/oriyuv';
    savepath = ['train_data_Y_','chromawithQP.h5'];

    QPvalue = QP/51;

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

        for i = 1 : length(filepaths)-test_num
        %for i = length(filepaths)-test_num+1 : length(filepaths)
            fprintf('yuv: %d\n',i);
            fid = fopen(fullfile(folder_data,filepaths(i).name),'rb');
            ImgData_Y = fread(fid, size_yuv(:,:,i), 'uint8');      % image data Y
            ImgData_Y = imresize(ImgData_Y,size_yuv(:,:,i)/2,'bicubic');
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

                    subim_input_Y = ImgData_Y(x : x+size_input-1, y : y+size_input-1);
                    subim_input_U = ImgData_U(x : x+size_input-1, y : y+size_input-1);
                    subim_input_V = ImgData_V(x : x+size_input-1, y : y+size_input-1);
                    subim_label_U = Imglabel_U(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
                    subim_label_V = Imglabel_V(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
                    count=count+1;
                    qp(:, :, 1, count) = QPvalue;
                    data1(:, :, 1, count) = subim_input_Y;
                    data2(:, :, 1, count) = subim_input_U;
                    data2(:, :, 2, count) = subim_input_V;
                    label(:, :, 1, count) = subim_label_U;
                    label(:, :, 2, count) = subim_label_V;
                end
            end
            fclose(fid);
            fclose(fid_label);
        end
end
tic
order = randperm(count);
qp = qp(:,:,1,order);
data1 = data1(:, :, :, order);
data2 = data2(:,:,:,order);
label = label(:, :, :, order);
toc
%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchqp = qp(:,:,:,last_read+1:last_read+chunksz);
    batchdata1 = data1(:,:,:,last_read+1:last_read+chunksz);
    batchdata2 = data2(:,:,:,last_read+1:last_read+chunksz);
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('qp',[1,1,1,totalct+1],'dat1',[1,1,1,totalct+1], 'dat2',[1,1,1,totalct+1],'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5_chromaQP(savepath, batchqp,batchdata1, batchdata2,batchlabs, ~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);