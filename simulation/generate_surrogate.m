function generate_surrogate(max_segments, input_dir, output_dir)
% RUN_BATCH_TIECKS_HIGHFS_ARMA  使用图片 ARMA 公式和 125 Hz 原始 ABP 生成 Tiecks 模拟 ABP-CBv 数据集。
%
% 本脚本是当前推荐使用的 Tiecks 数据生成主脚本。
%
% 核心思想：
%   旧流程是先把 ABP 从 125 Hz 降到 1 Hz，再把 1 Hz ABP 输入 Tiecks 模型。
%   这样在高 ARI 等级，尤其 ARI 8/9 时，快速自动调节动态可能已经在降采样前丢失。
%
%   当前流程改为：
%       1. 从原始 cuffless ABP 数据中读取 125 Hz ABP；
%       2. 在 125 Hz 下切成 100 秒窗口，步长 50 秒；
%       3. 对每个通过质量控制的 ABP 窗口，分别运行 ARI=0..9 的 Tiecks ARMA 递推公式；
%       4. 先得到 125 Hz 的相对 CBv 扰动，并在 125 Hz 上加入噪声；
%       5. 再将 ABP 与 CBv 使用同一套 0.5 Hz 低通 + 降采样流程同步降到 1 Hz；
%       6. 最终保存 N x 100 的相对 ABP/CBv 扰动样本，用于 CVAE 预训练。
%
% 用法：
%   generate_surrogate(100)   % 冒烟测试：只处理前 100 个有效 ABP 窗口
%   generate_surrogate        % 全量生成
%
% 输入：
%
% 输出：
%
% 输出变量：
%   dataset_abp      : M x 100 single，最终 1 Hz 相对 ABP 扰动片段
%   dataset_cbv      : M x 100 single，最终 1 Hz Tiecks 模拟相对 CBv 扰动片段
%   dataset_ari      : M x 1 uint8，每条样本对应的 ARI 标签 0..9
%   dataset_id       : M x 1 uint32，原始 case/record 编号
%   dataset_segment  : M x 1 uint32，同一 case 内第几个 100 秒窗口
%   meta             : 生成参数和统计信息
%
% 注意：
%   M = 有效 ABP 窗口数 x 10，因为每个 ABP 窗口都会生成 ARI=0..9 共 10 条 CBv。

% 如果没有指定 max_segments，则默认全量处理。
% max_segments 限制的是“有效 ABP 窗口数”，不是最终样本行数。
% 因为每个窗口会扩增为 10 个 ARI 样本，所以最终行数约为 max_segments * 10。
if nargin < 1 || isempty(max_segments)
    max_segments = inf;
end

clc;

% -------------------------------------------------------------------------
% 1. 路径设置
% -------------------------------------------------------------------------
% input_dir 中应包含原始 cuffless blood pressure 数据 Part_1.mat ~ Part_4.mat。
% 每个 Part 文件内部是 cell array，每个 cell 对应一条原始记录。
if nargin < 2 || isempty(input_dir)
    error('Specify input_dir containing the original Part_*.mat source files.');
end

% 默认输出到当前 tiecks 目录下的 results 文件夹。
if nargin < 3 || isempty(output_dir)
    output_dir = fullfile(fileparts(mfilename('fullpath')), 'results');
end
output_path = fullfile(output_dir, 'tiecks_simulated_data_highfs_arma_final.mat');

% -------------------------------------------------------------------------
% 2. 信号处理参数
% -------------------------------------------------------------------------
fs_raw = 125;        % 原始 ABP 采样率，单位 Hz
fs_target = 1;       % 最终样本采样率，单位 Hz
window_sec = 100;    % 每个窗口长度，100 秒
stride_sec = 50;     % 滑动窗口步长，50 秒，即 50% overlap
target_row = 2;      % 原始 matrix 中第 2 行为 ABP 通道
fc = 0.5;            % 抗混叠低通滤波截止频率，0.5 Hz
noise_fraction = 0.10; % 在 125 Hz CBv 上先加入 10% CBv 标准差的高斯噪声，再统一低通降采样
noise_seed = 42;       % 固定随机种子，保证重复生成结果可复现
rng(noise_seed, 'twister');

% 将秒数换算为原始采样点数。
window_len = window_sec * fs_raw;       % 100 * 125 = 12500 点
stride_len = stride_sec * fs_raw;       % 50  * 125 = 6250 点
downsample_factor = fs_raw / fs_target; % 125 Hz -> 1 Hz，降采样因子 125
target_len = window_sec * fs_target;    % 最终每条样本 100 点

% decimate 要求整数降采样因子。这里 125/1 正好为 125。
if abs(round(downsample_factor) - downsample_factor) > 1e-9
    error('fs_raw / fs_target must be an integer for this script.');
end
downsample_factor = round(downsample_factor);

% 检查输入/输出目录。
if ~isfolder(input_dir)
    error('Input folder not found: %s', input_dir);
end
if ~isfolder(output_dir)
    mkdir(output_dir);
end

% 读取所有原始 Part 文件。
file_list = dir(fullfile(input_dir, 'Part_*.mat'));
if isempty(file_list)
    error('No Part_*.mat files found in: %s', input_dir);
end

% 设计 3 阶 Butterworth 低通滤波器。
% 由于后续使用 filtfilt 前后向零相位滤波，实际幅频滚降相当于 6 阶。
% 这样既能抗混叠，又避免 4 阶 filtfilt 造成过强平滑。
[b_lp, a_lp] = butter(3, fc / (fs_raw / 2), 'low');

fprintf('[tiecks-highfs] input : %s\n', input_dir);
fprintf('[tiecks-highfs] output: %s\n', output_path);
fprintf('[tiecks-highfs] raw fs=%g Hz, target fs=%g Hz, window=%gs, stride=%gs\n', ...
    fs_raw, fs_target, window_sec, stride_sec);

% -------------------------------------------------------------------------
% 3. 预分配输出数组
% -------------------------------------------------------------------------
% 全量数据较大，为了避免每次循环动态增长导致速度很慢，先预分配一块空间。
% 如果后面行数超过 capacity，会调用 grow_storage 继续扩容。
capacity = 100000;
if isfinite(max_segments)
    capacity = max(10, min(capacity, max_segments * 10));
end

dataset_abp = zeros(capacity, target_len, 'single');
dataset_cbv = zeros(capacity, target_len, 'single');
dataset_ari = zeros(capacity, 1, 'uint8');
dataset_id = zeros(capacity, 1, 'uint32');
dataset_segment = zeros(capacity, 1, 'uint32');

row = 0;                         % 当前写到输出矩阵的第几行
valid_segments = 0;              % 通过 ABP 质量控制的 100 秒窗口数
rejected_segments = 0;           % 被质量控制或降采样拒绝的窗口数
global_case_id = uint32(0);      % 全局 case ID，每个 cell 对应一个 case
t_start = tic;
stop_now = false;

% -------------------------------------------------------------------------
% 4. 遍历 Part 文件、原始 case、100 秒窗口
% -------------------------------------------------------------------------
for fi = 1:numel(file_list)
    full_path = fullfile(file_list(fi).folder, file_list(fi).name);
    fprintf('\n[tiecks-highfs] loading %s (%d/%d)\n', file_list(fi).name, fi, numel(file_list));

    loaded = load(full_path);
    names = fieldnames(loaded);
    raw_cells = loaded.(names{1});
    if ~iscell(raw_cells)
        warning('Skipping %s: first variable is not a cell array.', file_list(fi).name);
        continue;
    end

    for ci = 1:numel(raw_cells)
        global_case_id = global_case_id + 1;
        current_matrix = raw_cells{ci};
        if size(current_matrix, 1) < target_row
            continue;
        end

        % 取第 2 行作为原始 ABP。此时仍是 125 Hz，没有降采样。
        abp_raw = double(current_matrix(target_row, :));
        data_len = numel(abp_raw);
        if data_len < window_len
            continue;
        end

        % 根据 100 秒窗口和 50 秒步长，计算该 case 可切出多少段。
        num_cuts = floor((data_len - window_len) / stride_len) + 1;
        for cut_idx = 0:(num_cuts - 1)
            % 冒烟测试模式下，达到指定有效窗口数后提前停止。
            if valid_segments >= max_segments
                stop_now = true;
                break;
            end

            start_idx = cut_idx * stride_len + 1;
            end_idx = start_idx + window_len - 1;
            abp_seg = abp_raw(start_idx:end_idx);  % 125 Hz, 12500 点

            % ABP 窗口质量控制：排除 NaN/Inf、常数信号、均值异常血压。
            if ~is_valid_abp_segment(abp_seg)
                rejected_segments = rejected_segments + 1;
                continue;
            end

            % 对输入 ABP 本身也走同一套低通 + 降采样流程，得到最终 1 Hz ABP 后再转相对扰动。
            % 注意：Tiecks 模型不是用这个 abp_out 运行，而是用上面的 125 Hz abp_seg 运行。
            abp_out = lowpass_decimate_1hz(abp_seg, b_lp, a_lp, downsample_factor, target_len);
            if isempty(abp_out)
                rejected_segments = rejected_segments + 1;
                continue;
            end

            % 与 Tiecks 内部输入 p=(ABP-mean)/mean 对齐：最终保存的 ABP 也使用相对扰动尺度。
            c_abp = mean(abp_seg);
            abp_out = (abp_out - c_abp) / c_abp;

            valid_segments = valid_segments + 1;

            % 对同一个 125 Hz ABP 窗口，分别生成 ARI=0..9 的 CBv。
            for ari = 0:9
                % 关键步骤：Tiecks ARMA 递推公式在 125 Hz 原始窗口的相对扰动上运行。
                % 这样保留高 ARI 下较快的动态变化。
                cbv_high = tiecks_filter_arma_formula(abp_seg, fs_raw, ari);

                % 先在 125 Hz Tiecks 输出上加入高斯噪声，噪声标准差为该条 CBv 标准差的 10%。
                % 然后再与 ABP 使用同一套低通 + 降采样流程同步降到 1 Hz。
                cbv_out = cbv_high;
                cbv_noise_std = noise_fraction * std(cbv_out);
                if cbv_noise_std > 0
                    cbv_out = cbv_out + cbv_noise_std * randn(size(cbv_out));
                end

                % Tiecks 输出也是 125 Hz，因此也需要与 ABP 同步低通并降到 1 Hz。
                cbv_out = lowpass_decimate_1hz(cbv_out, b_lp, a_lp, downsample_factor, target_len);
                if isempty(cbv_out)
                    error('Internal error: CBv decimation failed for case %d segment %d ARI %d.', ...
                        global_case_id, cut_idx + 1, ari);
                end

                row = row + 1;
                if row > size(dataset_abp, 1)
                    [dataset_abp, dataset_cbv, dataset_ari, dataset_id, dataset_segment] = grow_storage( ...
                        dataset_abp, dataset_cbv, dataset_ari, dataset_id, dataset_segment, target_len);
                end

                % 保存最终 1 Hz、100 点相对扰动样本。
                % 同一 case/segment 会连续保存 10 行，对应 ARI=0..9。
                dataset_abp(row, :) = single(abp_out);
                dataset_cbv(row, :) = single(cbv_out);
                dataset_ari(row) = uint8(ari);
                dataset_id(row) = global_case_id;
                dataset_segment(row) = uint32(cut_idx + 1);
            end

            if mod(valid_segments, 1000) == 0
                fprintf('[tiecks-highfs] valid windows=%d, rows=%d, elapsed=%.1fs\n', ...
                    valid_segments, row, toc(t_start));
            end
        end

        if stop_now
            break;
        end
    end

    if stop_now
        break;
    end
end

% 去掉预分配中没有用到的空行。
dataset_abp = dataset_abp(1:row, :);
dataset_cbv = dataset_cbv(1:row, :);
dataset_ari = dataset_ari(1:row);
dataset_id = dataset_id(1:row);
dataset_segment = dataset_segment(1:row);

% 保存元信息，方便之后确认数据生成口径。
meta = struct();
meta.source = input_dir;
meta.fs_raw = fs_raw;
meta.fs_target = fs_target;
meta.window_sec = window_sec;
meta.stride_sec = stride_sec;
meta.fc = fc;
meta.noise_fraction = noise_fraction;
meta.noise_seed = noise_seed;
meta.noise_note = 'Gaussian noise added to 125 Hz CBv before low-pass/downsampling; noise std = 10% of each high-fs CBv segment std';
meta.simulation_method = 'tiecks_arma_image_formula_highfs';
meta.valid_abp_windows = valid_segments;
meta.rejected_abp_windows = rejected_segments;
meta.tiecks_formula = 'V(z)/P(z)=(a+b*z^-1+c*z^-2)/(1+b*z^-1+c*z^-2)';
meta.tiecks_recurrence = 'v(n)=a*p(n)+b*(p(n-1)-v(n-1))+c*(p(n-2)-v(n-2))';
meta.tiecks_a_formula = 'a=1-K/Tz^2 (corrected; paper Eq.A10 printed (1-K/Tz)^2 contradicts Eq.A8)';
meta.tiecks_b_formula = 'b=1/Tz^2+2D/Tz-2';
meta.tiecks_c_formula = 'c=1-2D/Tz';
meta.tiecks_T_units = 'T table values are seconds; ARMA coefficients use Tz=T_seconds*fs_raw samples';
meta.tiecks_pressure_input = 'p=(ABP-mean(ABP))/mean(ABP)';
meta.dataset_abp_units = 'relative ABP perturbation after 1 Hz downsampling: (ABP_1Hz-mean(raw_window_ABP))/mean(raw_window_ABP)';
meta.dataset_cbv_units = 'relative CBv perturbation generated by Tiecks ARMA, with 125 Hz noise added before low-pass/downsampling';
meta.tiecks_filter_form = 'B=[a,b,c], A=[1,b,c] for ARI 1..9; B=1, A=1 for ARI 0; v=filter(B,A,p)';
meta.tiecks_initial_conditions = 'filter default zero history: p(n<=0)=0 and v(n<=0)=0';
meta.source_script = 'generate_surrogate.m';
meta.based_on = 'run_batch_tiecks_highfs.m';
meta.note = 'Tiecks simulated from raw 125 Hz relative ABP perturbation using the image ARMA recurrence. Final dataset_abp and dataset_cbv are both relative perturbations; CBv noise is added at 125 Hz before low-pass/downsampling.';

fprintf('\n[tiecks-highfs] generated %d rows from %d ABP windows\n', row, valid_segments);
fprintf('[tiecks-highfs] saving %s\n', output_path);
save(output_path, 'dataset_abp', 'dataset_cbv', 'dataset_ari', 'dataset_id', 'dataset_segment', 'meta', '-v7.3');

% 保存一张预览图，便于快速检查同一个 ABP 下 ARI 0..9 的 CBv 形态差异。
make_preview_plot(output_path, dataset_abp, dataset_cbv, dataset_ari, dataset_id);
fprintf('[tiecks-highfs] done. elapsed=%.1fs\n', toc(t_start));

end

function cbv_hat = tiecks_filter_arma_formula(abp, fs, ari)
% TIECKS_FILTER_ARMA_FORMULA  使用图片中的 Tiecks ARMA 递推公式实现。
%
% 输入：
%   abp : 1 x N 或 N x 1 的 ABP 序列，当前脚本中为 125 Hz、100 秒窗口
%   fs  : 采样率，当前为 125 Hz
%   ari : 自动调节指数，整数 0..9
% 输出：
%   cbv_hat : 归一化 CBv 输出。此处仍是 fs 采样率，后面再低通降采样。
%
% 数学逻辑：
%   先将 ABP 转成去均值压力扰动：
%       p = ABP - mean(ABP)
%   再根据 ARI 对应的 T, D, K 参数构造图片中的 ARMA 传递函数：
%       V(z) / P(z) = (a + b z^-1 + c z^-2) / (1 + b z^-1 + c z^-2)
%   并用 v = filter(B, A, p) 计算输出扰动。
%   由于本脚本在 125 Hz 原始信号上运行，Tiecks 表中的 T 秒会先换算为
%   采样点尺度 Tz = T * fs，再代入离散 z 域公式。

% 强制转为行向量，避免 MATLAB 列/行方向导致矩阵维度错误。
abp = double(abp(:))';

% 图片中的 P/V 公式不包含 CCP。这里仅去均值，得到压力扰动 p。
% 输入 ABP 保留 mmHg 尺度，后续滤波得到同尺度的 v 扰动。
c_abp = mean(abp);
p = (abp - c_abp)/c_abp;

if ari < 0 || ari > 9 || ari ~= round(ari)
    error('ARI must be an integer from 0 to 9.');
end

% ARI=0 表示完全被动。按要求令 B、A 全为 1，得到 v = p。
if ari == 0
    v=p;
    cbv_hat = v;
else
    % ARI=1..9 对应 Tiecks 表中的 T, D, K 参数。
    % 行号 ari=1 对应 params 第 1 行，ari=9 对应第 9 行。
    params = [ ...
        2.00, 1.60, 0.20; ...
        2.00, 1.50, 0.40; ...
        2.00, 1.15, 0.60; ...
        2.00, 0.90, 0.80; ...
        1.90, 0.75, 0.90; ...
        1.60, 0.65, 0.94; ...
        1.20, 0.55, 0.96; ...
        0.87, 0.52, 0.97; ...
        0.65, 0.50, 0.98];

    T = params(ari, 1);
    D = params(ari, 2);
    K = params(ari, 3);

    % 图片公式是离散 z 域传递函数，z^-1 对应一个采样点。
    % 因此需要把 Tiecks 表中的时间常数 T（秒）换算成采样点尺度 Tz。
    Tz = T * fs;

    % 论文 Appendix A 的 ARMA 系数（Panerai 2010, Eq. A10-A12）。
    % 注意：论文 Eq. A10 印刷为 a=(1-K/Tz)^2，与同篇 Eq. A8 分子常数项
    % (1-K/Tz^2) 自相矛盾。从 Eq. A5-A7 状态空间精确求解可证 a=1-K/Tz^2 才对，
    % 它使阶跃响应 DC 增益恒为 1-K（与状态空间稳态 2-K 的扰动量一致），且对任意 fs 稳定。
    % 用错误的 (1-K/Tz)^2 会让 DC 增益变成 1+K^2-2K*Tz，在高 fs 下爆炸。
    a = 1 - K / Tz^2;
    b = 1 / (Tz^2) + 2 * D / Tz - 2;
    c = 1 - 2 * D / Tz;

    B = [a, b, c];
    A = [1, b, c];

    % 等价于图片中的递推式：
    % v(n) = a*p(n) + b*(p(n-1)-v(n-1)) + c*(p(n-2)-v(n-2))。
    v = filter(B, A, p);
    cbv_hat = v;
end

end

function ok = is_valid_abp_segment(x)
% IS_VALID_ABP_SEGMENT  ABP 窗口质量控制。
%
% 当前只做基础 QC：
%   1. 全部为有限值；
%   2. 标准差不能接近 0，排除常数/坏信号；
%   3. 平均血压在 30~250 mmHg 之间。
ok = all(isfinite(x)) && std(x) > 1e-6 && mean(x) >= 30 && mean(x) <= 250;
end

function y = lowpass_decimate_1hz(x, b, a, factor, target_len)
% LOWPASS_DECIMATE_1HZ  将 125 Hz 信号低通滤波并降采样到 1 Hz。
%
% 输入 x 可以是 ABP，也可以是 Tiecks 生成的 CBv。
% 二者必须走同一套后处理流程，保证最终 ABP/CBv 时间对齐。
%
% 处理步骤：
%   1. 强制转为行向量；
%   2. 检查 NaN/Inf 和长度；
%   3. 滤波前去均值，减少 filtfilt 边缘偏移；
%   4. filtfilt 零相位低通；
%   5. 加回均值；
%   6. decimate 降采样；
%   7. 截取前 target_len=100 点。
x = double(x(:))';
if any(~isfinite(x)) || numel(x) < factor * target_len
    y = [];
    return;
end

x_mean = mean(x);
xf = filtfilt(b, a, x - x_mean) + x_mean;
y = decimate(xf, factor);

if numel(y) < target_len || any(~isfinite(y))
    y = [];
    return;
end
y = y(1:target_len);
end

function [abp, cbv, ari, id, seg] = grow_storage(abp, cbv, ari, id, seg, target_len)
% GROW_STORAGE  输出矩阵动态扩容。
%
% 全量生成时样本行数较多。如果预分配空间不够，就每次增加 extra 行。
extra = 100000;
abp = [abp; zeros(extra, target_len, 'single')];
cbv = [cbv; zeros(extra, target_len, 'single')];
ari = [ari; zeros(extra, 1, 'uint8')];
id = [id; zeros(extra, 1, 'uint32')];
seg = [seg; zeros(extra, 1, 'uint32')];
end

function make_preview_plot(output_path, dataset_abp, dataset_cbv, dataset_ari, dataset_id)
% MAKE_PREVIEW_PLOT  生成快速预览图。
%
% 预览图展示：
%   上图：某个窗口的输入 ABP；
%   下图：同一 ABP 下 ARI 0..9 生成的全部 CBv。
%
% 预期现象：
%   ARI 越高，CBv 对 ABP 波动的持续响应越弱，曲线通常更平稳。
if isempty(dataset_abp)
    return;
end

preview_path = strrep(output_path, '.mat', '_preview.png');
target_row = 1;

base_id = dataset_id(target_row);
same_case = find(dataset_id == base_id, 10, 'first');
if numel(same_case) < 10
    same_case = 1:min(10, size(dataset_abp, 1));
end

ari_indices = nan(1, 10);
for ari = 0:9
    idx = same_case(find(dataset_ari(same_case) == ari, 1));
    if ~isempty(idx)
        ari_indices(ari + 1) = idx;
    end
end

fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1100 700]);
subplot(2, 1, 1);
plot(dataset_abp(same_case(1), :), 'k', 'LineWidth', 1.2);
title(sprintf('Input ABP, case %d', base_id));
ylabel('Relative ABP');
grid on;
set(gca, 'Color', 'w');

subplot(2, 1, 2);
hold on;
colors = lines(10);
for ari = 0:9
    idx = ari_indices(ari + 1);
    if ~isnan(idx)
        plot(dataset_cbv(idx, :), 'Color', colors(ari + 1, :), 'LineWidth', 1.1, ...
            'DisplayName', sprintf('ARI %d', ari));
    end
end
title('Tiecks simulated CBv after 1 Hz downsampling');
ylabel('CBv normalized');
xlabel('Time (s)');
legend('Location', 'bestoutside');
grid on;
set(gca, 'Color', 'w');
hold off;

saveas(fig, preview_path);
close(fig);
fprintf('[tiecks-highfs] preview saved: %s\n', preview_path);
end
