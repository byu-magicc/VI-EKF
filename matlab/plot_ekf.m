function plot_ekf(name)

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
partial_update = read_line(config_file);
keyframe_reset = read_line(config_file);
keyframe_overlap = read_line(config_file);
num_features = read_line(config_file);
min_depth = read_line(config_file);

N = num_features;
len_xVector = 17+N*5;
len_dxVector = 16+N*3;


% Load data for plots
state_t = reshape(fread(fopen('/tmp/multirotor_true_state.log', 'r'), 'double'), 14, []); % [t;pos;att;vel;ang_vel]
imu_bias_drag_t = reshape(fread(fopen(strcat(['/tmp/multirotor_true_imu_biases_drag.log']), 'r'), 'double'), 8, []); % [t;acc_bias;gyro_bias;drag]
cmd_state_t = reshape(fread(fopen('/tmp/multirotor_commanded_state.log', 'r'), 'double'), 14, []); % [t;pos;att;vel;ang_vel]


figure()
set(gcf, 'name', 'Position', 'NumberTitle', 'off');
titles = ["x","y","z"];
idx = 1;
for i=1:3
    subplot(3, 1, i), hold on, grid on
    title(titles(i))
    plot(state_t(1,:), state_t(i + idx, :), 'linewidth', 2.0)
%     plot(ekf_state(1,:), ekf_state(i + idx, :), 'linewidth', 1.5)
    plot(cmd_state_t(1,:), cmd_state_t(i + idx, :), 'g--', 'linewidth', 1.0)
%     if plot_cov == true
%         plot(ekf_state(1,:), ekf_state(i + idx, :) + 2 * sqrt(ekf_cov(i + idx, :)), 'm-', 'linewidth', 0.5)
%         plot(ekf_state(1,:), ekf_state(i + idx, :) - 2 * sqrt(ekf_cov(i + idx, :)), 'm-', 'linewidth', 0.5)
%     end
    if i == 1
        legend('Truth', 'EKF', 'Command')
    end
end


figure()
set(gcf, 'name', 'Velocity', 'NumberTitle', 'off');
titles = ["x","y","z"];
idx = 8;
for i=1:3
    subplot(3, 1, i), hold on, grid on
    title(titles(i))
    plot(state_t(1,:), state_t(i + idx, :), 'linewidth', 2.0)
%     plot(ekf_state(1,:), ekf_state(i + idx, :), 'linewidth', 1.5)
%     if plot_cov == true
%         plot(ekf_state(1,:), ekf_state(i + idx, :) + 2 * sqrt(ekf_cov(i + idx, :)), 'm-', 'linewidth', 0.5)
%         plot(ekf_state(1,:), ekf_state(i + idx, :) - 2 * sqrt(ekf_cov(i + idx, :)), 'm-', 'linewidth', 0.5)
%     end
    if i == 1
        legend('Truth', 'EKF')
    end
end


figure()
set(gcf, 'name', 'Attitude', 'NumberTitle', 'off');
titles = ["w","x","y","z"];
idx = 4;
for i=1:4
    subplot(4, 1, i), hold on, grid on
    title(titles(i))
    plot(state_t(1,:), state_t(i + idx, :), 'linewidth', 2.0)
%     plot(ekf_state(1,:), ekf_state(i + idx, :), 'linewidth', 1.5)
%     if plot_cov == true
%         plot(ekf_state(1,:), ekf_state(i + idx, :) + 2 * sqrt(ekf_cov(i + idx, :)), 'm-', 'linewidth', 0.5)
%         plot(ekf_state(1,:), ekf_state(i + idx, :) - 2 * sqrt(ekf_cov(i + idx, :)), 'm-', 'linewidth', 0.5)
%     end
    if i == 1
        legend('Truth', 'EKF')
    end
end


figure()
set(gcf, 'name', 'Accel Bias', 'NumberTitle', 'off');
titles = ["x","y","z"];
idx = 1;
for i=1:3
    subplot(3, 1, i), hold on, grid on
    title(titles(i))
    plot(imu_bias_drag_t(1,:), imu_bias_drag_t(i + idx, :), 'linewidth', 2.0)
%     plot(ekf_state(1,:), ekf_state(i + idx, :), 'linewidth', 1.5)
%     if plot_cov == true
%         plot(ekf_state(1,:), ekf_state(i + idx, :) + 2 * sqrt(ekf_cov(i + idx, :)), 'm-', 'linewidth', 0.5)
%         plot(ekf_state(1,:), ekf_state(i + idx, :) - 2 * sqrt(ekf_cov(i + idx, :)), 'm-', 'linewidth', 0.5)
%     end
    if i == 1
        legend('Truth', 'EKF')
    end
end


figure()
set(gcf, 'name', 'Gyro Bias', 'NumberTitle', 'off');
titles = ["x","y","z"];
idx = 4;
for i=1:3
    subplot(3, 1, i), hold on, grid on
    title(titles(i))
    plot(imu_bias_drag_t(1,:), imu_bias_drag_t(i + idx, :), 'linewidth', 2.0)
%     plot(ekf_state(1,:), ekf_state(i + idx, :), 'linewidth', 1.5)
%     if plot_cov == true
%         plot(ekf_state(1,:), ekf_state(i + idx, :) + 2 * sqrt(ekf_cov(i + idx, :)), 'm-', 'linewidth', 0.5)
%         plot(ekf_state(1,:), ekf_state(i + idx, :) - 2 * sqrt(ekf_cov(i + idx, :)), 'm-', 'linewidth', 0.5)
%     end
    if i == 1
        legend('Truth', 'EKF')
    end
end


figure()
set(gcf, 'name', 'Drag Coeff', 'NumberTitle', 'off');
hold on, grid on
title("Drag Coefficient")
idx = 7;
plot(imu_bias_drag_t(1,:), imu_bias_drag_t(1 + idx, :), 'linewidth', 2.0)
%     plot(ekf_state(1,:), ekf_state(i + idx, :), 'linewidth', 1.5)
%     if plot_cov == true
%         plot(ekf_state(1,:), ekf_state(i + idx, :) + 2 * sqrt(ekf_cov(i + idx, :)), 'm-', 'linewidth', 0.5)
%         plot(ekf_state(1,:), ekf_state(i + idx, :) - 2 * sqrt(ekf_cov(i + idx, :)), 'm-', 'linewidth', 0.5)
%     end
legend('Truth', 'EKF')


% % Plot drag
% figure(6); clf;
% set(gcf, 'name', 'Drag', 'NumberTitle', 'off');
% for i = 1:2
%     plot(xhat(1,:), xhat(18,:),'lineWidth', 3.0);
%     title('drag');
% end
