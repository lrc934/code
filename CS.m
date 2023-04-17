function [curve, bestx,bestf]=CS(fobj, varargin)
global optimizer   % 获得提供的优化算法
popsize = cell2mat(varargin(1));   % 种群大小
maxgen = cell2mat(varargin(2));   % 最大迭代次数
lb = cell2mat(varargin(3));   % 变量下界
ub = cell2mat(varargin(4));   % 变量上界
dim = cell2mat(varargin(5));  % 变量个数
% Discovery rate of alien eggs/solutions
pa=0.4;
% 随机初始解
nest=randn(popsize,dim);
fbest=ones(popsize,1)*10^(100); % 最小化问题
Kbest=1;
bestx = zeros(1, dim);
bestf = inf;
for j=1:maxgen
    % 找到当前最好的
    Kbest=get_best_nest(fbest);
    % 选择一个随机的巢(避免当前最好的)
    k=choose_a_nest(popsize,Kbest);
    bestnest=nest(Kbest,:);
    % 生成新的解决方案 (但保持当前最好的)
    s=get_a_cuckoo(nest(k,:),bestnest);
    % 范围约束
    s = max(min(s, ub), lb);
    % 评估这个解决方案
    fnew=fobj(s);
    if fnew<=fbest(k)
        fbest(k)=fnew;
        nest(k,:)=s;
    end
    % 发现和随机化
    if rand<pa
        k=get_max_nest(fbest);
        s=emptyit(nest(k,:));
        nest(k,:)=s;
        s = max(min(s,ub),lb);
        fbest(k)=fobj(s);
    end

    %% 找到当前最好的和显示
    [fval,I]=min(fbest);
    bestsol=nest(I,:);
    %% 更新全局最优个体和适应度
    if fval < bestf
        bestf = fval;
        bestx =bestsol;
    end
    curve(j) = bestf;
end

if ~isempty(cell2mat(varargin(6))), clc; print_copr;
    model_name = evalin('base', 'model_name');
    disp(model_name);   % 打印模型信息
    disp('running... ...')
    disp('End Of Run... ...')
end

%_______________________________________________________________

