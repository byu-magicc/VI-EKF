function plot_ekf(name, plot_cov)

 % Load configuration data
config_file = fopen(strcat(['/tmp/',name,'_config.txt']), 'r');
fgetl(config_file); % Test Num:
x0 = read_line(config_file);
P0 = read_line(config_file);
P0_feat = read_line(config_file);
Qx = read_line(config_file);
Qx_feat = read_line(config_file);
Qu = read_line(config_file);
q_b_c = read_line(config_file);
p_b_c = read_line(config_file);
lambda = read_line(config_file);
labmda_feat = read_line(config_file);
use_partial_update = read_line(config_file);
use_keyframe_reset = read_line(config_file);
use_drag_term = read_line(config_file);
keyframe_overlap = read_line(config_file);
num_features = read_line(config_file);
min_depth = read_line(config_file);


% Load data for plots
state_t = reshape(fread(fopen('/tmp/multirotor_true_state.log', 'r'), 'double'), 14, []); % [t;pos;att;vel;ang_vel]
imu_bias_drag_t = reshape(fread(fopen('/tmp/multirotor_true_imu_biases_drag.log', 'r'), 'double'), 8, []); % [t;acc_bias;gyro_bias;drag]
cmd_state_t = reshape(fread(fopen('/tmp/multirotor_commanded_state.log', 'r'), 'double'), 14, []); % [t;pos;att;vel;ang_vel]
state_e = reshape(fread(fopen(strcat(['/tmp/',name,'_state.log']), 'r'), 'double'), 1+17+num_features*5, []); % [t;pos;vel;att;ba;bg;drag;q_feat;rho;...]
cov_e = reshape(fread(fopen(strcat(['/tmp/',name,'_cov.log']), 'r'), 'double'), 1+16+num_features*3, []); % [t;pos;vel;att;ba;bg;drag;q_feat;rho;...]
global_pose_e = reshape(fread(fopen(strcat(['/tmp/',name,'_global_pose.log']), 'r'), 'double'), 8, []); % [t;pos;att]
imu = reshape(fread(fopen(strcat(['/tmp/',name,'_input.log']), 'r'), 'double'), 7, []); % [t;acc;gyro]


figure()
set(gcf, 'name', 'Position', 'NumberTitle', 'off');
titles = ["x","y","z"];
for i=1:3
    subplot(3, 1, i), hold on, grid on
    title(titles(i))
    plot(state_t(1,:), state_t(i + 1, :), 'linewidth', 2.0)
    plot(global_pose_e(1,:), global_pose_e(i + 1, :), 'r', 'linewidth', 1.5)
    plot(cmd_state_t(1,:), cmd_state_t(i + 1, :), 'g--', 'linewidth', 1.0)
    if plot_cov == true
        plot(cov_e(1,:), global_pose_e(i + 1, :) + 2 * sqrt(cov_e(i + 1, :)), 'm-', 'linewidth', 0.5)
        plot(cov_e(1,:), global_pose_e(i + 1, :) - 2 * sqrt(cov_e(i + 1, :)), 'm-', 'linewidth', 0.5)
    end
    if i == 1
        legend('Truth', 'EKF', 'Command')
    end
end


figure()
set(gcf, 'name', 'Velocity', 'NumberTitle', 'off');
titles = ["x","y","z"];
for i=1:3
    subplot(3, 1, i), hold on, grid on
    title(titles(i))
    plot(state_t(1,:), state_t(i + 8, :), 'linewidth', 2.0)
    plot(state_e(1,:), state_e(i + 4, :), 'r', 'linewidth', 1.5)
    if plot_cov == true
        plot(cov_e(1,:), state_e(i + 4, :) + 2 * sqrt(cov_e(i + 4, :)), 'm-', 'linewidth', 0.5)
        plot(cov_e(1,:), state_e(i + 4, :) - 2 * sqrt(cov_e(i + 4, :)), 'm-', 'linewidth', 0.5)
    end
    if i == 1
        legend('Truth', 'EKF')
    end
end


figure()
set(gcf, 'name', 'Attitude', 'NumberTitle', 'off');
titles = ["w","x","y","z"];
for i=1:4
    subplot(4, 1, i), hold on, grid on
    title(titles(i))
    plot(state_t(1,:), state_t(i + 4, :), 'linewidth', 2.0)
    plot(global_pose_e(1,:), global_pose_e(i + 4, :), 'r', 'linewidth', 1.5)
    if plot_cov == true
        plot(cov_e(1,:), global_pose_e(i + 4, :) + 2 * sqrt(cov_e(i + 7, :)), 'm-', 'linewidth', 0.5)
        plot(cov_e(1,:), global_pose_e(i + 4, :) - 2 * sqrt(cov_e(i + 7, :)), 'm-', 'linewidth', 0.5)
    end
    if i == 1
        legend('Truth', 'EKF')
    end
end


figure()
set(gcf, 'name', 'Accel Bias', 'NumberTitle', 'off');
titles = ["x","y","z"];
for i=1:3
    subplot(3, 1, i), hold on, grid on
    title(titles(i))
    plot(imu_bias_drag_t(1,:), imu_bias_drag_t(i + 1, :), 'linewidth', 2.0)
    plot(state_e(1,:), state_e(i + 11, :), 'r', 'linewidth', 1.5)
    if plot_cov == true
        plot(cov_e(1,:), state_e(i + 11, :) + 2 * sqrt(cov_e(i + 10, :)), 'm-', 'linewidth', 0.5)
        plot(cov_e(1,:), state_e(i + 11, :) - 2 * sqrt(cov_e(i + 10, :)), 'm-', 'linewidth', 0.5)
    end
    if i == 1
        legend('Truth', 'EKF')
    end
end


figure()
set(gcf, 'name', 'Gyro Bias', 'NumberTitle', 'off');
titles = ["x","y","z"];
for i=1:3
    subplot(3, 1, i), hold on, grid on
    title(titles(i))
    plot(imu_bias_drag_t(1,:), imu_bias_drag_t(i + 4, :), 'linewidth', 2.0)
    plot(state_e(1,:), state_e(i + 14, :), 'r', 'linewidth', 1.5)
    if plot_cov == true
        plot(cov_e(1,:), state_e(i + 14, :) + 2 * sqrt(cov_e(i + 13, :)), 'm-', 'linewidth', 0.5)
        plot(cov_e(1,:), state_e(i + 14, :) - 2 * sqrt(cov_e(i + 13, :)), 'm-', 'linewidth', 0.5)
    end
    if i == 1
        legend('Truth', 'EKF')
    end
end


figure()
set(gcf, 'name', 'Drag Coeff', 'NumberTitle', 'off');
hold on, grid on
title("Drag Coefficient")
plot(imu_bias_drag_t(1,:), imu_bias_drag_t(1 + 7, :), 'linewidth', 2.0)
plot(state_e(1,:), state_e(1 + 17, :), 'r', 'linewidth', 1.5)
if plot_cov == true
    plot(cov_e(1,:), state_e(1 + 17, :) + 2 * sqrt(cov_e(1 + 16, :)), 'm-', 'linewidth', 0.5)
    plot(cov_e(1,:), state_e(1 + 17, :) - 2 * sqrt(cov_e(1 + 16, :)), 'm-', 'linewidth', 0.5)
end
legend('Truth', 'EKF')


figure()
set(gcf, 'name', 'Accelerometer', 'NumberTitle', 'off');
titles = ["x","y","z"];
for i=1:3
    subplot(3, 1, i), hold on, grid on
    title(titles(i))
    plot(imu(1,:), imu(i + 1, :), 'linewidth', 2.0)
end


figure()
set(gcf, 'name', 'Rate Gyro', 'NumberTitle', 'off');
titles = ["x","y","z"];
for i=1:3
    subplot(3, 1, i), hold on, grid on
    title(titles(i))
    plot(imu(1,:), imu(i + 4, :), 'linewidth', 2.0)
end
