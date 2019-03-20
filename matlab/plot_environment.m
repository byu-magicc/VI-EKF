function plot_environment(name)

% Load data
lm = reshape(fread(fopen('/tmp/landmarks.log', 'r'), 'double'), 3, []); % [north;east;down]
state_t = reshape(fread(fopen('/tmp/multirotor_true_state.log', 'r'), 'double'), 14, []); % [t;pos;att;vel;ang_vel]
cmd_state_t = reshape(fread(fopen('/tmp/multirotor_commanded_state.log', 'r'), 'double'), 14, []); % [t;pos;att;vel;ang_vel]


% Plot 3D scene
figure(), hold on, grid on
set(gcf, 'name', 'Environment', 'NumberTitle', 'off')
set(gca, 'YDir', 'reverse')
set(gca, 'ZDir', 'reverse')
set(gcf, 'color', 'w')
title('Environment')
plot3([0, 1], [0, 0], [0, 0], 'r','HandleVisibility','off')
plot3([0, 0], [0, 1], [0, 0], 'b','HandleVisibility','off')
plot3([0, 0], [0, 0], [0, 1], 'g','HandleVisibility','off')
plot3(lm(1,:), lm(2,:), lm(3,:), 'k.', 'MarkerSize', 2.0,'HandleVisibility','off')
plot3(state_t(2,:), state_t(3,:), state_t(4,:), 'b', 'linewidth', 1.5)
% plot3(air_est(2,:), air_est(3,:), air_est(4,:), 'r', 'linewidth', 1.5)
plot3(cmd_state_t(2,:), cmd_state_t(3,:), cmd_state_t(4,:), 'g--', 'linewidth', 1.0)
view(-50, 20)
axis equal
xlabel('North')
ylabel('East')
zlabel('Down')
legend('Multirotor Truth','Multirotor Est','Multirotor Command')
