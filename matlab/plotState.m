format compact
set(0,'DefaultFigureWindowStyle','docked')

logTopDir = dir('../logs');
subFolders = logTopDir([logTopDir.isdir]);
dirNames = {subFolders.name};
latestDir = dirNames{end};
logDir = strcat(strcat('../logs/', latestDir), '/');

names = ["t", "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "ax", "ay", "az", "bx", "by", "bz"];

 %% Configuration
file = fopen(strcat(logDir, 'conf.txt'), 'r');
fgetl(file); % Test Num:
x0 = read_line(file);
P0 = read_line(file);
P0_feat = read_line(file);
Qx = read_line(file);
Qx_feat = read_line(file);
Qu = read_line(file);
q_b_c = read_line(file);
p_b_c = read_line(file);
lambda = read_line(file);
labmda_feat = read_line(file);
partial_update = read_line(file);
keyframe_reset = read_line(file);
drag_term = read_line(file);
keyframe_overlap = read_line(file);
num_features = read_line(file);
min_depth = read_line(file);

N = num_features;
len_xVector = 17+N*5;
len_dxVector = 16+N*3;


%% Read files
est_file = fopen(strcat(logDir, 'prop.bin'), 'r');
pos_file = fopen(strcat(logDir, 'POS.bin'), 'r');
att_file = fopen(strcat(logDir, 'ATT.bin'), 'r');
imu_file = fopen(strcat(logDir, 'input.bin'), 'r');

% Read state
prop_data = reshape(fread(est_file, 'double'), (1+len_xVector+len_dxVector), []);
xhat = prop_data(1:len_xVector+1,:);
Pdiag = prop_data(len_xVector+2:end,:);

% Read Position Measurements
pos = reshape(fread(pos_file, 'double'), 1+6+1, []);
att = reshape(fread(att_file, 'double'), 1+8+1, []);

% Read IMU
imu = reshape(fread(imu_file, 'double'), 1+6, []);



%% Plot States
%% Plot position
figure(1); clf;
set(gcf, 'name', 'Position', 'NumberTitle', 'off');
for i = 1:3
    idx = i+1;
    subplot(3,1,i);
    plot(pos(1,:), pos(idx,:), '-', 'lineWidth', 3);
    hold on;
    plot(xhat(1,:), xhat(idx,:), '-', 'lineWidth', 2.0);
    title(names(idx));
    legend("mocap", "est")
end

%% Plot Velocity
figure(2); clf;
set(gcf, 'name', 'Velocity', 'NumberTitle', 'off');
for i = 1:3
    idx = i+4;
    subplot(3,1,i);
    plot(xhat(1,:), xhat(idx,:), '-', 'lineWidth', 2.0);
    title(names(idx));
    legend("est")
end

%% Plot Attitude
figure(3); clf;
set(gcf, 'name', 'Attitude', 'NumberTitle', 'off');
for i = 1:4
    idx = i+7;
    subplot(4,1,i);
    plot(att(1,:), att(1+i,:), 'lineWidth', 3.0);
    hold on
    plot(xhat(1,:), xhat(idx,:), '-', 'lineWidth', 2.0);
    title(names(idx));
    legend("mocap","est")
end

%% Plot Biases
figure(4); clf;
set(gcf, 'name', 'Biases', 'NumberTitle', 'off');
for i = 1:2
    for j = 1:3
        idx = (i-1)*3+j+11;
        subplot(3,2,(j-1)*2+i)
        plot(xhat(1,:), xhat(idx,:),'lineWidth', 3.0);
        title(names(idx));
    end
end

%% Plot drag
figure(6); clf;
set(gcf, 'name', 'Drag', 'NumberTitle', 'off');
for i = 1:2
    plot(xhat(1,:), xhat(18,:),'lineWidth', 3.0);
    title('drag');
end


%% Plot Inputs
figure(5); clf;
set(gcf, 'name', 'Input', 'NumberTitle', 'off');
for i = 1:2
    for j = 1:3
        imu_idx = (i-1)*3+j+1;
        subplot(3,2,(j-1)*2+i)
        plot(imu(1,:), imu(imu_idx,:),'lineWidth', 3.0);
        title(names(imu_idx+10));
    end
end
