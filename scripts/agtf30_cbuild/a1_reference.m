function a1_reference(nSec)
%A1_REFERENCE  Normal vs Rapid Accelerator on the same flight, every logged signal.
%   Cases: healthy and one degraded setting (DEGRADED below). For each case the
%   package is set up like the campaign worker (setup_simulation, carriers,
%   define_inputs with ICPoint auto -> steady-state IC, compressor-map logging
%   routes), then the same MWS is simulated in Normal and in Rapid Accelerator
%   mode. All root outputs, To Workspace variables and logsout Dataset elements
%   are flattened (cb_flatten) and compared signal by signal (cb_compare).
%   Writes to %AGTF30_WORK%\agtf30_cbuild\a1\: ref_<case>.mat (Normal signals: names,
%   time, data cells), mws_<case>.mat, compare_<case>.csv, a1_summary.json.
if nargin < 1, nSec = 1800; end
E = cb_env();
outDir = fullfile(E.root, 'a1'); if ~exist(outDir, 'dir'), mkdir(outDir); end
cases = {'healthy', struct('mws_path', {}, 'theta', {}); 'degraded', degraded()};
R = struct('flight_seconds', nSec, 'csv', E.csv, 'cases', struct());
for c = 1:size(cases, 1)
    nm = cases{c, 1};
    fprintf('=== case %s ===\n', nm);
    [MWS, F] = cb_prepare(E, nSec, cases{c, 2});
    save(fullfile(outDir, ['mws_' nm '.mat']), 'MWS', 'F');
    load_system(fullfile(E.engDir, [E.model '.mdl']));
    routes = agtf30_stage1_map_evidence('instrument', E.model); %#ok<NASGU>
    args = {'StopTime', num2str(F.t(end)), 'ReturnWorkspaceOutputs', 'on', ...
        'LimitDataPoints', 'off', 'SignalLogging', 'on'};
    tic; outN = sim(E.model, 'SimulationMode', 'normal', args{:}); tN = toc;
    fprintf('normal %.1f s\n', tN);
    SN = cb_flatten(outN); clear outN
    ref = SN; ref.case = nm; ref.mode = 'normal'; %#ok<STRNU>
    save(fullfile(outDir, ['ref_' nm '.mat']), '-struct', 'ref', '-v7.3');
    routes = agtf30_stage1_map_evidence('instrument', E.model); %#ok<NASGU>
    tic; outR = sim(E.model, 'SimulationMode', 'rapid-accelerator', args{:}); tR = toc;
    fprintf('rapid %.1f s (incl. any build)\n', tR);
    SR = cb_flatten(outR); clear outR
    T = cb_compare(SN, SR, fullfile(outDir, ['compare_' nm '.csv']));
    k = strcmp(T.status, 'compared');
    s = struct('normal_s', tN, 'rapid_s', tR, 'signals_ref', numel(SN.names), ...
        'signals_test', numel(SR.names), 'compared', nnz(k), ...
        'columns_compared', sum(T.cols(k)), 'not_compared', {T.signal(~k)'}, ...
        'status_not_compared', {T.status(~k)'}, ...
        'max_abs', max(T.max_abs(k)), 'max_rel', max(T.max_rel(k)), ...
        'max_rel_pointwise', max(T.max_rel_pointwise(k)), ...
        'max_time_abs', max(T.time_max_abs(k)), 'nonfinite_mismatch', sum(T.nonfinite_mismatch), ...
        'n_identical', nnz(k & T.max_abs == 0), 'n_rel_gt_1e6', nnz(k & T.max_rel > 1e-6));
    [~, iw] = max(T.max_rel .* k);
    s.worst_signal = T.signal{iw};
    R.cases.(nm) = s;
    disp(s)
    clear SN SR ref
end
R.degraded = degraded();
fid = fopen(fullfile(outDir, 'a1_summary.json'), 'w');
fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);
fprintf('A1_DONE\n');
end

function th = degraded()
% one degraded health setting: HPT efficiency -2 %, HPC efficiency -1 %, HPC flow -1.5 %
th = struct('mws_path', {'MWS.HPT.s_Eff', 'MWS.HPC.s_Eff', 'MWS.HPC.s_Wc'}, ...
    'theta', {-0.02, -0.01, -0.015});
end
