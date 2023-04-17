function [curve, bestx,bestf]=CS(fobj, varargin)
global optimizer   % ����ṩ���Ż��㷨
popsize = cell2mat(varargin(1));   % ��Ⱥ��С
maxgen = cell2mat(varargin(2));   % ����������
lb = cell2mat(varargin(3));   % �����½�
ub = cell2mat(varargin(4));   % �����Ͻ�
dim = cell2mat(varargin(5));  % ��������
% Discovery rate of alien eggs/solutions
pa=0.4;
% �����ʼ��
nest=randn(popsize,dim);
fbest=ones(popsize,1)*10^(100); % ��С������
Kbest=1;
bestx = zeros(1, dim);
bestf = inf;
for j=1:maxgen
    % �ҵ���ǰ��õ�
    Kbest=get_best_nest(fbest);
    % ѡ��һ������ĳ�(���⵱ǰ��õ�)
    k=choose_a_nest(popsize,Kbest);
    bestnest=nest(Kbest,:);
    % �����µĽ������ (�����ֵ�ǰ��õ�)
    s=get_a_cuckoo(nest(k,:),bestnest);
    % ��ΧԼ��
    s = max(min(s, ub), lb);
    % ��������������
    fnew=fobj(s);
    if fnew<=fbest(k)
        fbest(k)=fnew;
        nest(k,:)=s;
    end
    % ���ֺ������
    if rand<pa
        k=get_max_nest(fbest);
        s=emptyit(nest(k,:));
        nest(k,:)=s;
        s = max(min(s,ub),lb);
        fbest(k)=fobj(s);
    end

    %% �ҵ���ǰ��õĺ���ʾ
    [fval,I]=min(fbest);
    bestsol=nest(I,:);
    %% ����ȫ�����Ÿ������Ӧ��
    if fval < bestf
        bestf = fval;
        bestx =bestsol;
    end
    curve(j) = bestf;
end

if ~isempty(cell2mat(varargin(6))), clc; print_copr;
    model_name = evalin('base', 'model_name');
    disp(model_name);   % ��ӡģ����Ϣ
    disp('running... ...')
    disp('End Of Run... ...')
end

%_______________________________________________________________

