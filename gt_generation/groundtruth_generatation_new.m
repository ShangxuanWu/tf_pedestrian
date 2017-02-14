% Generate heatmaps for pedestrian images based on location of multiple
% joints.


% COMMENTS:
% 1) check visibility of joints before making heatmaps
% 2) Visibility based on 2 aspects:
% - pixel location within bounds
% - no huge overlap between pedestrians (use depth coordinate too)
% 3) Width of Gaussian can be determined by depth or by distance between
% multiple joints (e.g. head and center or head and feet or head and
% lowerneck)
%-------------------------------------------------------------------------
%
% TODO generate smaller heatmaps here

% Shangxuan Wu - Comments:
% generate a 13-channel mat file with the segment-gt incorporated.
% original images are bmp files
% original segmentation files are bmp files with instance-level 

clc;
clear all;
tic;
%
joint_size = 13;
root_dir = '/mnt/sdc1/shangxuan/dataset/synthetic1';
dest_dir = '/mnt/sdc1/shangxuan/dataset/synthetic1';

    % folders
    image_dir = [root_dir, '/image'];
    annotation_dir = [root_dir, '/joint_coordinate'];
    segmentation_dir = [root_dir, '/depthmap'];
    dest_dir = [root_dir, '/gt'];

    % joints
    %annotation_joints = {'center', 'feet', 'head', 'upperneck', 'leftfoot', 'rightfoot', 'leftknee', 'rightknee', 'lowerneck', 'righthand', 'lefthand', 'highercenter'};
    annotation_joints = {'Right_ankle', 'Right_knee', 'Right_hip', 'Left_hip', 'Left_knee', 'Left_ankle', 'Right_wrist', 'Right_elbow', 'Right_shoulder', 'Left_shoulder', 'Left_elbow', 'Left_wrist', 'Neck_Head', 'top'};

    % parameters
    %jointscales = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}; %scaling for gaussians
    jointHeatmapCount = 14;

    imfiles = dir([image_dir, '/*.png']); %get list of all images in directory

    % read through all the files
    for j = 1:length(imfiles)
        j
        updateText = ['file ', num2str(j), ' out of ', num2str(length(imfiles))]

        % pure name
        %pure_name_without_extension = sprintf('%05d',j-1);
        pure_name_without_extension = imfiles(j).name(1:5);
        % read image
        image_full_name = [image_dir, '/', imfiles(j).name];
        im = imread(image_full_name);
        % don't know the original size of pic
        [H,W,~] = size(im);
        
        % read joints
        joint_full_name = [annotation_dir, '/joint_', pure_name_without_extension, '.txt'];
        fileID = fopen(joint_full_name, 'r');
        lines = textscan(fileID, '%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d ','Delimiter',' ');
        fclose(fileID);

        % start rendering joints
        heatmap = zeros(H, W, jointHeatmapCount + 1); % tensorflow format: N H W C
        
        for l = 1:jointHeatmapCount
            this_channel = zeros([H,W]);
            for k = 1:size(lines{1})
                
                joint_x = lines{2*l-1}(k);
                joint_y = lines{2*l}(k);
                
                if(joint_x > 0 && joint_y > 0 && joint_x <= W && joint_y <= H)
                    if joint_x - joint_size > 0
                        joint_x_start = joint_x - joint_size;
                    else
                        joint_x_start = 1;
                    end
                    if joint_y - joint_size > 0
                        joint_y_start = joint_y - joint_size;
                    else
                        joint_y_start = 1;
                    end
                    if joint_x + joint_size <= W
                        joint_x_end = joint_x + joint_size;
                    else
                        joint_x_end = W;
                    end
                    if joint_y + joint_size <= H
                        joint_y_end = joint_y + joint_size;
                    else
                        joint_y_end = H;
                    end
                    
                    this_channel(joint_y_start:joint_y_end,joint_x_start:joint_x_end) = 1;
                end
            end
            
            heatmap(:,:,l+1) = this_channel;
        end
        
        % read segmentation
        seg_full_name = [segmentation_dir, '/', imfiles(j).name];        
        seg = imread(seg_full_name);
        seg_sum = sum(seg, 3);
        segmask = (seg_sum > 0);
        heatmap(:,:,1) = segmask;

        % resize total heatmap
        heatmap = imresize(heatmap, [288, 384]);

        % normalize to 0~1
        for j = 1:jointHeatmapCount + 1
            max_ = max(max(max(heatmap(:,:,j))));
            min_ = min(min(min(heatmap(:,:,j))));
            range_ = max_ - min_;
            if range_ ~= 0
                heatmap(:,:,j) = (heatmap(:,:,j) - min_)/range_;
            end
        end 
    
        save_fn = [dest_dir,'/', pure_name_without_extension,'.mat'];
        save(save_fn, 'heatmap', '-v6');
    end

    toc