% 假设数据集中时间数据在第7列
time_column = 7;
data_columns = 1:6; % 前6列为数据列

% 第一个数据集
data_ur3 = out.robot1_joint_vel.Data;
time_ur3 = data_ur3(:, time_column);

% 第二个数据集
data_ur3e = out.robot2_joint_vel.Data;
time_ur3e = data_ur3e(:, time_column);

% 在第一个数据集中找到第一个非零数据行
first_nonzero_row_ur3 = find(any(out.start.Data(:, 1) ~= 0, 2), 1, 'first');
start_time_ur3 = data_ur3(first_nonzero_row_ur3, time_column);
end_time_ur3 = start_time_ur3 + 30;  % 计算15秒后的结束时间

% 在第二个数据集中找到与第一个数据集相同或最接近的时间戳
[~, idx] = min(abs(time_ur3e - start_time_ur3));
start_time_ur3e = time_ur3e(idx);
end_time_ur3e = start_time_ur3e + 30;  % 计算15秒后的结束时间

% 选取两个数据集从这些时间点开始，15秒内的数据
valid_rows_ur3 = time_ur3 >= start_time_ur3 & time_ur3 <= end_time_ur3;
valid_rows_ur3e = time_ur3e >= start_time_ur3e & time_ur3e <= end_time_ur3e;

valid_data_ur3 = data_ur3(valid_rows_ur3, data_columns);
valid_data_ur3e = data_ur3e(valid_rows_ur3e, data_columns);

% 计算两个数据集每个对应元素的平方和
% 需要确保两个数据集的行数相同
min_rows = min(size(valid_data_ur3, 1), size(valid_data_ur3e, 1));
combined_squares = valid_data_ur3(1:min_rows, :).^2 + valid_data_ur3e(1:min_rows, :).^2;

% 对平方和开根号
combined_roots = sqrt(combined_squares);

% 对每一行的结果求和
sum_sqrt = sum(combined_roots, 2);

% 使用trapz进行时间积分
% 确保时间向量的长度与数据长度相同
time_vector = linspace(start_time_ur3, end_time_ur3, min_rows);
integral_result = trapz(time_vector, sum_sqrt);

% 输出积分结果
disp(integral_result);
