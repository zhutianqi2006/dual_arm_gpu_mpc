h_fig = figure('Name', 'ur3_joint_vel_plot');
time_line = 7;

% Find the index where any of the first 6 columns is non-zero
non_zero_index = find(any(out.ur3_joint_vel.Data(:,1:6) ~= 0, 2), 1, 'first');

% Adjusting data to start plotting from the first non-zero index
time_data = out.ur3_joint_vel.Data(non_zero_index:end, time_line);
time_start  = out.ur3_joint_vel.Data(non_zero_index, time_line);
data_columns = out.ur3_joint_vel.Data(non_zero_index:end, 1:6);

% Plotting each joint velocity
for i = 1:6
    plot(time_data-time_start, data_columns(:,i), 'LineWidth', 2.2, 'LineStyle', '-.');
    hold on;
end

% Setting plot parameters
set(gca, 'XLim', [0 15], 'FontSize', 16);
set(gca, 'YLim', [-0.7, 0.7], 'FontSize', 16);
set(gca, 'YTick', -0.7:0.2:0.7);
grid on;
legend({'$joint1_v$', '$joint2_v$', '$joint3_v$', '$joint4_v$', '$joint5_v$', '$joint6_v$'}, ...
       'FontSize', 16, 'Interpreter', 'latex', 'NumColumns', 2);
xlabel('Time (s)', 'FontSize', 20, 'Interpreter', 'tex');
ylabel('Velocity (rad/s)', 'FontSize', 20, 'Interpreter', 'tex');

% Save the figure
saveas(h_fig, h_fig.Name, 'fig');
saveas(h_fig, h_fig.Name, 'svg');