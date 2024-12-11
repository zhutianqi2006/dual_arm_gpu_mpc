h_fig = figure('Name', 'dual_arm_abs_feedback_plot');
time_line = 9;

% Find the index where any of the first 8 columns is non-zero
non_zero_index = find(any(out.abs_feedback.Data(:,1:8) ~= 0, 2), 1, 'first');

% Adjusting data to start plotting from the first non-zero index
time_data = out.abs_feedback.Data(non_zero_index:end, time_line);
time_start  = out.abs_feedback.Data(non_zero_index, time_line);
data_columns = out.abs_feedback.Data(non_zero_index:end, 1:8);

% Plotting each error
for i = 1:8
    plot(time_data-time_start, data_columns(:,i), 'LineWidth', 2.2, 'LineStyle', '-.');
    hold on;
end

% Setting plot parameters
set(gca, 'XLim', [0 15], 'FontSize', 16);
set(gca, 'YLim', [-0.1, 0.2], 'FontSize', 16);
set(gca, 'YTick', -0.3:0.05:0.3);
grid on;
legend({'$error_1$', '$error_2$', '$error_3$', '$error_4$', '$error_5$', '$error_6$', '$error_7$', '$error_8$'}, ...
       'FontSize', 16, 'Interpreter', 'latex', 'NumColumns', 2);
xlabel('Time (s)', 'FontSize', 20, 'Interpreter', 'tex');
ylabel('Error', 'FontSize', 20, 'Interpreter', 'tex');

% Save the figure
saveas(h_fig, h_fig.Name, 'fig');
saveas(h_fig, h_fig.Name, 'svg');