%% ��ʼ������
warning off         % �رձ�����Ϣ
close all               % �ر�����ͼ��
clear                    % ��ձ���
clc                        % ���������
%print_copr;           

%% ��ȡ����
data = xlsread('����3.xlsx');

%% ����ѵ��������Լ�
%temp = randperm(318);
temp = 1:1:318;
inputTrainDataset = data(temp(1:265), 1:7)';  
outputTrainDataset = data(temp(1:265), 8)';   

inputTestDataset = data(temp(265:318), 1:7)'; 
outputTestDataset = data(temp(265:318), 8)';  

%% ��һ������
[inputn_train, input_ps] = mapminmax(inputTrainDataset, 0, 1);
inputn_test = mapminmax('apply', inputTestDataset, input_ps);
[outputn_train, output_ps] = mapminmax(outputTrainDataset, 0, 1);

%% �����Ż��㷨
global optimizer   % ����ṩ���Ż��㷨
model_name = sprintf('��ǰ���е���, Comparison diagram of predicted and actual values of.', func2str(optimizer));
disp(model_name);   % ��ӡģ����Ϣ
disp('running... ...')
tic;
maxgen=2000;   
popsize=318;   
dim=2;    
lb = [1, 1]; 
ub = [100, 10]; 
fobj=@(x)func(x,inputn_train,outputn_train);  
[curve, optimized_param]=optimizer(fobj, popsize,maxgen,lb,ub,dim, 1);  
% �����Ż��㷨�Ľ�������
figure
plot(curve, 'r-', 'LineWidth', 1.0)
grid on
xlabel('Evolutionary algebra')
ylabel('Best fitness')
title('Evolution curve')

%% ʹ���Ż���Ĳ�������ģ��
%��ȡ����������
ntree = round(optimized_param(1));   
mtry = round(optimized_param(2));   
extra_options.DEBUG_ON=0; 

%% ѵ��ģ��
model = regRF_train(inputn_train',outputn_train',ntree,mtry,extra_options);

%% ʶ��ͷ���һ��
model_out1 = regRF_predict(inputn_train',model);    
model_out2 =  regRF_predict(inputn_test',model);     
predictTrainDataset = mapminmax('reverse', model_out1', output_ps);  
predictTestDataset = mapminmax('reverse', model_out2', output_ps);    
toc;

%% �������
disp('ѵ��������������: ')
MSE = mean((outputTrainDataset - predictTrainDataset).^2);
disp(['�������MSE = ', num2str(MSE)])
MAE = mean(abs(outputTrainDataset - predictTrainDataset));
disp(['ƽ���������MAE = ', num2str(MAE)])
RMSE = sqrt(MSE);
disp(['���������RMSE = ', num2str(RMSE)])
MAPE = mean(abs((outputTrainDataset - predictTrainDataset)./outputTrainDataset));
disp(['ƽ�����԰ٷֱ����MAPE = ', num2str(MAPE*100), '%'])
R = corrcoef(outputTrainDataset, predictTrainDataset);
R2 = R(1, 2)^2;
disp(['����Ŷ�R2 = ', num2str(R2)])
disp(' ')
disp('���Լ�����������: ')
MSE_test = mean((outputTestDataset - predictTestDataset).^2);
disp(['�������MSE = ', num2str(MSE_test)])
MAE_test = mean(abs(outputTestDataset - predictTestDataset));
disp(['ƽ���������MAE = ', num2str(MAE_test)])
RMSE_test = sqrt(MSE_test);
disp(['���������RMSE = ', num2str(RMSE_test)])
MAPE_test = mean(abs((outputTestDataset - predictTestDataset)./outputTestDataset));
disp(['ƽ�����԰ٷֱ����MAPE = ', num2str(MAPE_test*100), '%'])
R_test = corrcoef(outputTestDataset, predictTestDataset);
R2_test = R_test(1, 2)^2;
disp(['����Ŷ�R2 = ', num2str(R2_test)])

%% �Խ����ͼ
% ѵ����
figure
plot(outputTrainDataset, 'b*-', 'LineWidth', 0.8)
hold on
plot(predictTrainDataset, 'ro-', 'LineWidth', 0.8)
grid on
xlabel('ѵ���������')
ylabel('bulk Modulus/MPa')
legend('ʵ��ֵ', 'Ԥ��ֵ')
title({strcat(model_name(8:end - 5), 'ѵ����Ԥ��ֵ��ʵ��ֵ�Ա�ͼ'), ['RMSE = ', num2str(RMSE) , ' R2 = ', num2str(R2)]})

figure
plot(outputTrainDataset - predictTrainDataset, 'b*-', 'LineWidth', 0.8)
grid on
xlabel('ѵ���������')
ylabel('Ԥ��ƫ��')
legend('���')
title({strcat(model_name(8:end - 5), 'ѵ����Ԥ�����ͼ'), ['ƽ�����԰ٷֱ����MAPE = ', num2str(MAPE*100), '%']})

% ���Լ�
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
xlabel('�����������')
ylabel('Ԥ��ƫ��')
legend('���')
title({strcat(model_name(8:end - 5), '���Լ�Ԥ�����ͼ'), ['ƽ�����԰ٷֱ����MAPE = ', num2str(MAPE_test*100), '%']})



