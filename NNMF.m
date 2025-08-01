clc
clear
%读取相关文件
data = xlsread("C:\Users\19317\Desktop\肌肉协同\非负矩阵分解\data1.xlsx"); %#ok<XLSRD>
Wfinal = []; 
Hfinal = []; 
VAFfinal = []; 
Dfinal = [];
%转置
data = data';
number =8;
%为了克服法陷入局部极小值的问题,每次重复都是使用不同的初始条件（初始矩阵 W0 和 H0）。
%算法会尝试 10 次，每次迭代100次，并返回其中损失函数最小的一组 W 和 H 作为最终的输出结果
opt = statset('MaxIter', 200, 'Display', 'final','TolFun',1e-4);
[W0,H0] = nnmf(data,number,'Replicates',20,...
                   'Options',opt,...
                   'Algorithm','als');

ires = W0*H0;
%Continue with more iterations from the best of these results using alternating least squares.
opt = statset('Maxiter',200,'Display','final','TolFun',1e-4);
[W,H] = nnmf(data,number,'W0',W0,'H0',H0,...
                 'Options',opt,...
                 'Algorithm','mult');

res = W*H;
%Total Variance Accounted For (𝑉𝐴𝐹)
VAF = 1-norm(data-W*H,'fro')/norm(data,'fro');

%The Variance Accounted For (𝑉𝐴𝐹) computed for each of the observed muscles
num_samples = size(data, 1);
vaf_per_row = zeros(1, num_samples);
for i = 1:num_samples
    % 提取第 i 行数据和对应的重构结果
    data_row = data(i, :);
    reconstructed_row = res(i, :);

    % 计算每一行的 VAF
    vaf_per_row(i) = 1 - norm(data_row - reconstructed_row, 'fro') / norm(data_row,'fro');
end


D = norm(data-W*H,'fro')/sqrt(size(data,1)*size(data,2));

W1=W./repmat(max(max(W)),[size(W)]);%生成一个与矩阵 W 相同大小的矩阵，其中所有元素都是 W 中的最大元素,将矩阵 W 中的每个元素除以相应位置上的最大元素，实现了列归一化
H1=H./repmat(max(H,[],2),[1,size(H,2)]); %同理

%fres = W1*H1;

Wfinal = [Wfinal,W1]; 
Hfinal = [Hfinal;H1]; 
VAFfinal = [VAFfinal,VAF];
Dfinal = [Dfinal,D];