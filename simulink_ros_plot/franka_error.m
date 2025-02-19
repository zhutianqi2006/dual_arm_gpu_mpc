% 假设data是你的数据矩阵
time_column = 9; % 时间数据在第9列
data_columns = 1:8; % 前7列为数据列

% 查找第一个非零数据行，即前7列至少有一个非零
first_nonzero_row = find(any(out.start.Data(:, 1) ~= 0, 2), 1, 'first');
start_time = out.abs_error.Data(first_nonzero_row, time_column); % 获取有效数据开始的时间
end_time = start_time + 20.0; % 计算结束时间

% 找到在有效时间范围内的数据行
valid_rows = out.abs_error.Data(:, time_column) >= start_time & out.abs_error.Data(:, time_column) <= end_time;

% 提取这些行的前8列
valid_data = out.abs_error.Data(valid_rows, data_columns);

% 计算每行的平方和
squared_sums = sum(valid_data.^2, 2);

% 计算每行平方和的平方根
root_sums = sqrt(squared_sums);

% 获取对应的时间向量
time_vector = out.abs_error.Data(valid_rows, time_column);

% 使用trapz进行时间积分
integral_result = trapz(time_vector, root_sums);

% 输出积分结果
disp(integral_result);
