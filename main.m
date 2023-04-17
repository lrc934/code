%% 初始化程序
warning off         % 关闭报警信息
close all               % 关闭所有图窗
clear                    % 清空变量
clc                        % 清空命令行
%print_copr;           

%% 读取数据
data = xlsread('数据3.xlsx');

%% 划分训练集与测试集
%temp = randperm(318);
temp = 1:1:318;
inputTrainDataset = data(temp(1:265), 1:7)';  
outputTrainDataset = data(temp(1:265), 8)';   

inputTestDataset = data(temp(265:318), 1:7)'; 
outputTestDataset = data(temp(265:318), 8)';  

%% 归一化处理
[inputn_train, input_ps] = mapminmax(inputTrainDataset, 0, 1);
inputn_test = mapminmax('apply', inputTestDataset, input_ps);
[outputn_train, output_ps] = mapminmax(outputTrainDataset, 0, 1);

%% 调用优化算法
global optimizer   % 获得提供的优化算法
model_name = sprintf('当前运行的是, Comparison diagram of predicted and actual values of.', func2str(optimizer));
disp(model_name);   % 打印模型信息
disp('running... ...')
tic;
maxgen=2000;   
popsize=318;   
dim=2;    
lb = [1, 1]; 
ub = [100, 10]; 
fobj=@(x)func(x,inputn_train,outputn_train);  
[curve, optimized_param]=optimizer(fobj, popsize,maxgen,lb,ub,dim, 1);  
% 绘制优化算法的进化曲线
figure
plot(curve, 'r-', 'LineWidth', 1.0)
grid on
xlabel('Evolutionary algebra')
ylabel('Best fitness')
title('Evolution curve')

%% 使用优化后的参数赋给模型
%提取超参数变量
ntree = round(optimized_param(1));   
mtry = round(optimized_param(2));   
extra_options.DEBUG_ON=0; 

%% 训练模型
model = regRF_train(inputn_train',outputn_train',ntree,mtry,extra_options);

%% 识别和反归一化
model_out1 = regRF_predict(inputn_train',model);    
model_out2 =  regRF_predict(inputn_test',model);     
predictTrainDataset = mapminmax('reverse', model_out1', output_ps);  
predictTestDataset = mapminmax('reverse', model_out2', output_ps);    
toc;

%% 分析误差
disp('训练集误差计算如下: ')
MSE = mean((outputTrainDataset - predictTrainDataset).^2);
disp(['均方误差MSE = ', num2str(MSE)])
MAE = mean(abs(outputTrainDataset - predictTrainDataset));
disp(['平均绝对误差MAE = ', num2str(MAE)])
RMSE = sqrt(MSE);
disp(['根均方误差RMSE = ', num2str(RMSE)])
MAPE = mean(abs((outputTrainDataset - predictTrainDataset)./outputTrainDataset));
disp(['平均绝对百分比误差MAPE = ', num2str(MAPE*100), '%'])
R = corrcoef(outputTrainDataset, predictTrainDataset);
R2 = R(1, 2)^2;
disp(['拟合优度R2 = ', num2str(R2)])
disp(' ')
disp('测试集误差计算如下: ')
MSE_test = mean((outputTestDataset - predictTestDataset).^2);
disp(['均方误差MSE = ', num2str(MSE_test)])
MAE_test = mean(abs(outputTestDataset - predictTestDataset));
disp(['平均绝对误差MAE = ', num2str(MAE_test)])
RMSE_test = sqrt(MSE_test);
disp(['根均方误差RMSE = ', num2str(RMSE_test)])
MAPE_test = mean(abs((outputTestDataset - predictTestDataset)./outputTestDataset));
disp(['平均绝对百分比误差MAPE = ', num2str(MAPE_test*100), '%'])
R_test = corrcoef(outputTestDataset, predictTestDataset);
R2_test = R_test(1, 2)^2;
disp(['拟合优度R2 = ', num2str(R2_test)])

%% 对结果作图
% 训练集
figure
plot(outputTrainDataset, 'b*-', 'LineWidth', 0.8)
hold on
plot(predictTrainDataset, 'ro-', 'LineWidth', 0.8)
grid on
xlabel('训练样本序号')
ylabel('bulk Modulus/MPa')
legend('实际值', '预测值')
title({strcat(model_name(8:end - 5), '训练集预测值和实际值对比图'), ['RMSE = ', num2str(RMSE) , ' R2 = ', num2str(R2)]})

figure
plot(outputTrainDataset - predictTrainDataset, 'b*-', 'LineWidth', 0.8)
grid on
xlabel('训练样本序号')
ylabel('预测偏差')
legend('误差')
title({strcat(model_name(8:end - 5), '训练集预测误差图'), ['平均绝对百分比误差MAPE = ', num2str(MAPE*100), '%']})

% 测试集
figure
plot(outputTestDataset, 'b*-', 'LineWidth', 0.8)
hold on
plot(predictTestDataset, 'ro-', 'LineWidth', 0.8)
grid on
xlabel('Samples')
ylabel('bulk Modulus/GPa')
legend('Measured value', 'Predicted value')
title({strcat(model_name(8:end - 5))},{' random forest test set optimized by Cuckoo search algorithm', ['RMSE = ', num2str(RMSE_test), '  R^2 = ', num2str(R2_test)]})

figure
plot(outputTestDataset - predictTestDataset, 'b*-', 'LineWidth', 0.8)
grid on
xlabel('测试样本序号')
ylabel('预测偏差')
legend('误差')
title({strcat(model_name(8:end - 5), '测试集预测误差图'), ['平均绝对百分比误差MAPE = ', num2str(MAPE_test*100), '%']})



