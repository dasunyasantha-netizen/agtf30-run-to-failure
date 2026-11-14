function info = cb_make_models(NMAX, pkgName, ssExtra)
%CB_MAKE_MODELS  Code-generation copies of AGTF30SysDyn / AGTF30SysSS / AGTF30_eng.
%   Works on the private copy %AGTF30_WORK%\agtf30_cbuild\pkgc (never the original).
%   Edits (saved in pkgc only; equations, solver and controller unchanged):
%     * every block parameter that carries a per-run value is pointed at a
%       global Simulink.Parameter CB_* (storage class ExportedGlobal), so the
%       generated C has a plain global the runner sets before initialization:
%         - health parameters  MWS.<FAN|LPC|HPC|HPT|LPT>.s_<Nc|Wc|PR|Eff>  (AGTF30_eng)
%         - Dyn ICs            MWS.In.{ICcom,N2IC,N3IC,WfIC,VAFNIC,VBVIC,Sensor.IC*}
%         - flight inputs      MWS.In.{Alt,MN,dT,PLA} (time row and value row) as
%                              fixed-length 1 x NMAX vectors (flight padded with
%                              its last value at t_end+1, t_end+2, ...; values on
%                              [0, t_end] are unchanged)
%         - SS inputs          MWS.In.{AltIC,MNIC,dTambIC,ssNc1,VAFNIC,VBVIC} and
%                              [ICcom ICss]' (start guess)
%     * the constant manual-request profiles (N1creq, Wfreq, VBVreq, VAFNreq; their
%       value is the same at every time) become two-point constant tables;
%     * one root Outport per model (cb_y) carries the dataset channels selected
%       from the out_Dyn / out_SS bus (order in info.dyn_outputs / ss_outputs);
%     * code generation: grt.tlc, default parameter behaviour Inlined (MWS
%       constants folded), no MAT-file logging, generate code only, packNGo.
%   Writes pkgc\AGTF30\cb_params.mat (CB_* objects) and cb_models.json.
%   Optional (phase B): pkgName (private copy, default 'pkgc') and ssExtra, true to
%   append ssExtraSignals() from the SS 'Engine Model' bus (stall margins, T45,
%   speeds, thrust) after the unchanged out_SS channels.
if nargin < 1 || isempty(NMAX), NMAX = 30000; end
if nargin < 2 || isempty(pkgName), pkgName = 'pkgc'; end
if nargin < 3, ssExtra = false; end
E = cb_env([], pkgName);
[MWS, F] = cb_prepare(E, 1800);            % healthy, speed-test flight: build-time values
close_system('AGTF30SysSS', 0); close_system('AGTF30_eng', 0);
info = struct('NMAX', NMAX, 'edits', struct('model', {}, 'block', {}, 'param', {}, 'old', {}, 'new', {}));

% ---------- per-run parameter map: expression -> CB name, value
P = struct();
comps = {'FAN', 'LPC', 'HPC', 'HPT', 'LPT'}; sc = {'Nc', 'Wc', 'PR', 'Eff'};
engMap = {};
for c = comps
    for s = sc
        nm = sprintf('CB_%s_s_%s', c{1}, s{1});
        engMap(end+1, :) = {sprintf('MWS.%s.s_%s', c{1}, s{1}), nm}; %#ok<AGROW>
        P.(nm) = MWS.(c{1}).(['s_' s{1}]);
    end
end
dynMap = {'MWS.In.WfIC', 'CB_WfIC'; 'MWS.In.VAFNIC', 'CB_VAFNIC'; 'MWS.In.VBVIC', 'CB_VBVIC'; ...
    'MWS.In.ICcom', 'CB_ICcom'; 'MWS.In.N2IC', 'CB_N2IC'; 'MWS.In.N3IC', 'CB_N3IC'};
P.CB_WfIC = MWS.In.WfIC; P.CB_VAFNIC = MWS.In.VAFNIC; P.CB_VBVIC = MWS.In.VBVIC;
P.CB_ICcom = MWS.In.ICcom; P.CB_N2IC = MWS.In.N2IC; P.CB_N3IC = MWS.In.N3IC;
for f = fieldnames(MWS.In.Sensor)'
    if startsWith(f{1}, 'IC')
        dynMap(end+1, :) = {['MWS.In.Sensor.' f{1}], ['CB_Sensor_' f{1}]}; %#ok<AGROW>
        P.(['CB_Sensor_' f{1}]) = MWS.In.Sensor.(f{1});
    end
end
n = numel(F.t);
P.CB_In_t = [F.t, F.t(end) + (1:NMAX - n)];
for v = {'Alt', 'MN', 'dT', 'PLA'}
    x = MWS.In.(v{1});
    if ~isequal(x(2, :), F.t), error('cb:inputs', 'MWS.In.%s time row differs from t', v{1}); end
    dynMap(end+1, :) = {sprintf('MWS.In.%s(2,:)', v{1}), 'CB_In_t'}; %#ok<AGROW>
    dynMap(end+1, :) = {sprintf('MWS.In.%s(1,:)', v{1}), ['CB_In_' v{1}]}; %#ok<AGROW>
    P.(['CB_In_' v{1}]) = [x(1, :), repmat(x(1, end), 1, NMAX - n)];
end
for v = {'N1creq', 'Wfreq', 'VBVreq', 'VAFNreq'}
    x = MWS.In.(v{1});
    if any(x(1, :) ~= x(1, 1)), error('cb:const', 'MWS.In.%s is not constant', v{1}); end
    dynMap(end+1, :) = {sprintf('MWS.In.%s(2,:)', v{1}), '[0 1e9]'}; %#ok<AGROW>
    dynMap(end+1, :) = {sprintf('MWS.In.%s(1,:)', v{1}), sprintf('[%s %s]', ...
        num2str(x(1, 1), 17), num2str(x(1, 1), 17))}; %#ok<AGROW>
end
ssMap = {'MWS.In.ssNc1', 'CB_ssNc1'; 'MWS.In.VAFNIC', 'CB_VAFNIC'; 'MWS.In.VBVIC', 'CB_VBVIC'; ...
    'MWS.In.AltIC', 'CB_AltIC'; 'MWS.In.MNIC', 'CB_MNIC'; 'MWS.In.dTambIC', 'CB_dTambIC'; ...
    '[MWS.In.ICcom MWS.In.ICss]''', 'CB_SSIC'};
P.CB_ssNc1 = MWS.In.ssNc1; P.CB_AltIC = MWS.In.AltIC; P.CB_MNIC = MWS.In.MNIC;
P.CB_dTambIC = MWS.In.dTambIC; P.CB_SSIC = [MWS.In.ICcom MWS.In.ICss]';

% ---------- Simulink.Parameter objects (ExportedGlobal doubles, fixed sizes)
CB = struct();
for f = fieldnames(P)'
    p = Simulink.Parameter(double(P.(f{1})));
    p.CoderInfo.StorageClass = 'ExportedGlobal';
    p.DataType = 'double';
    CB.(f{1}) = p;
    assignin('base', f{1}, p);
end
save(fullfile(E.engDir, 'cb_params.mat'), '-struct', 'CB');
info.params = structfun(@(p) size(p.Value), CB, 'UniformOutput', false);

% ---------- edit the three models
cd(E.engDir);
load_system('AGTF30_eng'); load_system('AGTF30SysDyn'); load_system('AGTF30SysSS');
info.edits = [replaceParams('AGTF30_eng', engMap), replaceParams('AGTF30SysDyn', dynMap), ...
    replaceParams('AGTF30SysSS', ssMap)];
info.unreplaced = [leftovers('AGTF30_eng', engMap), leftovers('AGTF30SysDyn', dynMap), ...
    leftovers('AGTF30SysSS', ssMap)];
set_param('AGTF30SysDyn', 'StopTime', 'inf');

% root outputs
info.dyn_outputs = addRootOut('AGTF30SysDyn', 'out_Dyn', dynSignals());
if ssExtra
    info.ss_outputs = addRootOut('AGTF30SysSS', 'out_SS', ssSignals(), ...
        'AGTF30SysSS/Engine Model', ssExtraSignals());
else
    info.ss_outputs = addRootOut('AGTF30SysSS', 'out_SS', ssSignals());
end

% code generation settings
for m = {'AGTF30_eng', 'AGTF30SysDyn', 'AGTF30SysSS'}
    cs = getActiveConfigSet(m{1});
    set_param(cs, 'SystemTargetFile', 'grt.tlc');
    set_param(cs, 'DefaultParameterBehavior', 'Inlined');
    set_param(cs, 'MatFileLogging', 'off');
    set_param(cs, 'GenCodeOnly', 'on');
    set_param(cs, 'GenerateReport', 'off');
    set_param(cs, 'SupportNonFinite', 'on');
    set_param(cs, 'SupportContinuousTime', 'on');
    set_param(cs, 'SupportNonInlinedSFcns', 'on');
    set_param(cs, 'GRTInterface', 'off');
    set_param(cs, 'CodeInterfacePackaging', 'Nonreusable function');
    set_param(cs, 'RTWVerbose', 'off');
    if ~strcmp(m{1}, 'AGTF30_eng')
        set_param(cs, 'PackageGeneratedCodeAndArtifacts', 'on');
        set_param(cs, 'PackageName', [m{1} '_cb.zip']);
    end
end
save_system('AGTF30_eng'); save_system('AGTF30SysDyn'); save_system('AGTF30SysSS');
fid = fopen(fullfile(E.engDir, 'cb_models.json'), 'w');
fprintf(fid, '%s', jsonencode(info, 'PrettyPrint', true)); fclose(fid);
fprintf('CB_MAKE_MODELS_DONE edits=%d unreplaced=%d\n', numel(info.edits), numel(info.unreplaced));
end

function ed = replaceParams(model, map)
ed = struct('model', {}, 'block', {}, 'param', {}, 'old', {}, 'new', {});
blks = find_system(model, 'LookUnderMasks', 'all', 'FollowLinks', 'off', 'Type', 'block');
for i = 1:numel(blks)
    dp = get_param(blks{i}, 'DialogParameters');
    if ~isstruct(dp), continue; end
    for f = fieldnames(dp)'
        try v = get_param(blks{i}, f{1}); catch, continue; end
        if ~ischar(v), continue; end
        k = find(strcmp(strtrim(v), map(:, 1)), 1);
        if isempty(k), continue; end
        set_param(blks{i}, f{1}, map{k, 2});
        ed(end+1) = struct('model', model, 'block', blks{i}, 'param', f{1}, 'old', v, 'new', map{k, 2}); %#ok<AGROW>
    end
end
end

function L = leftovers(model, map)
% block parameters that still mention a mapped MWS field in another form
L = {};
blks = find_system(model, 'LookUnderMasks', 'all', 'FollowLinks', 'off', 'Type', 'block');
keys = regexprep(map(:, 1), '\(.*$', '');
keys = unique(regexprep(keys, '^\[', ''));
for i = 1:numel(blks)
    dp = get_param(blks{i}, 'DialogParameters');
    if ~isstruct(dp), continue; end
    for f = fieldnames(dp)'
        try v = get_param(blks{i}, f{1}); catch, continue; end
        if ischar(v) && any(cellfun(@(k) contains(v, k), keys))
            L{end+1} = sprintf('%s:%s=%s', blks{i}, f{1}, v); %#ok<AGROW>
        end
    end
end
end

function names = addRootOut(model, wsVar, sigs, src2Block, sigs2)
% Bus Selector on the signal feeding the To Workspace block 'wsVar' -> Mux -> Outport cb_y
% (optional second Bus Selector on output 1 of src2Block, appended after sigs)
if nargin < 4, src2Block = ''; sigs2 = {}; end
tw = find_system(model, 'LookUnderMasks', 'all', 'BlockType', 'ToWorkspace', 'VariableName', wsVar);
if numel(tw) ~= 1, error('cb:tw', '%s: %d To Workspace blocks for %s', model, numel(tw), wsVar); end
ph = get_param(tw{1}, 'PortHandles');
ln = get_param(ph.Inport(1), 'Line');
src = get_param(ln, 'SrcPortHandle');
parent = get_param(tw{1}, 'Parent');
if ~strcmp(parent, model)
    error('cb:twParent', '%s To Workspace %s is not at the top level (%s)', model, wsVar, parent);
end
bs = add_block('simulink/Signal Routing/Bus Selector', [model '/cb_sel'], 'Position', [100 900 140 1300]);
set_param(bs, 'OutputSignals', strjoin(sigs, ','));
nTot = numel(sigs) + numel(sigs2);
mx = add_block('simulink/Signal Routing/Mux', [model '/cb_mux'], 'Inputs', num2str(nTot), ...
    'Position', [300 900 305 1300]);
op = add_block('simulink/Sinks/Out1', [model '/cb_y'], 'Position', [400 1090 430 1110]);
bph = get_param(bs, 'PortHandles');
add_line(model, src, bph.Inport(1));
mph = get_param(mx, 'PortHandles');
for k = 1:numel(sigs)   % every channel as double (e.g. solver iteration counts are integers)
    dc = add_block('simulink/Signal Attributes/Data Type Conversion', sprintf('%s/cb_dbl%d', model, k), ...
        'OutDataTypeStr', 'double', 'Position', [200 900+10*k 230 908+10*k]);
    dph = get_param(dc, 'PortHandles');
    add_line(model, bph.Outport(k), dph.Inport(1));
    add_line(model, dph.Outport(1), mph.Inport(k));
end
if ~isempty(sigs2)
    bs2 = add_block('simulink/Signal Routing/Bus Selector', [model '/cb_sel2'], 'Position', [100 1400 140 1800]);
    set_param(bs2, 'OutputSignals', strjoin(sigs2, ','));
    sph = get_param(src2Block, 'PortHandles');
    b2ph = get_param(bs2, 'PortHandles');
    add_line(model, sph.Outport(1), b2ph.Inport(1));
    for k = 1:numel(sigs2)
        dc = add_block('simulink/Signal Attributes/Data Type Conversion', sprintf('%s/cb_dbl2_%d', model, k), ...
            'OutDataTypeStr', 'double', 'Position', [200 1400+10*k 230 1408+10*k]);
        dph = get_param(dc, 'PortHandles');
        add_line(model, b2ph.Outport(k), dph.Inport(1));
        add_line(model, dph.Outport(1), mph.Inport(numel(sigs) + k));
    end
end
% zero-order hold at the base rate: the generated step function then reports the
% values of the major step (what Simulink logs), not those of the last ode4 stage
zh = add_block('simulink/Discrete/Zero-Order Hold', [model '/cb_zoh'], 'SampleTime', ...
    get_param(model, 'FixedStep'), 'Position', [340 1090 370 1110]);
zph = get_param(zh, 'PortHandles');
add_line(model, mph.Outport(1), zph.Inport(1));
oph = get_param(op, 'PortHandles');
add_line(model, zph.Outport(1), oph.Inport(1));
names = [sigs(:); sigs2(:)]';
end

function s = dynSignals()
s = {'eng.S2.Tt', 'eng.S24.Tt', 'eng.S36.Tt', 'eng.S48.Tt', 'eng.S5.Tt', 'eng.S4.Tt', 'eng.S45.Tt', ...
    'eng.S17.Pt', 'eng.S2.Pt', 'eng.S21.Pt', 'eng.S24.Pt', 'eng.S36.Ps', 'eng.S36.Pt', 'eng.S4.Pt', ...
    'eng.S45.Pt', 'eng.S5.Pt', 'eng.Shaft.N_Fan', 'eng.Shaft.N_LPC', 'eng.Shaft.N_HPC', ...
    'eng.S21.W', 'eng.S24.W', 'eng.S25.W', 'eng.S48.W', 'eng.S5.W', ...
    'eng.SM.SMFan', 'eng.SM.SMLPC', 'eng.SM.SMHPC', 'eng.Perf.Fnet', 'eng.Perf.TSFC', ...
    'eng.Data.HPC_Data.Wbleeds', 'act.Wfact', 'act.VBVact', 'act.VAFNact', 'cntrl.RegNum', ...
    'in.Alt', 'in.MN', 'in.dT', 'in.PLA', ...
    'loop.Solver.Iterations', 'loop.Solver.Dependents', 'loop.Solver.Independents', ...
    'eng.Data.FAN_Data.NcMap', 'eng.Data.LPC_Data.NcMap', 'eng.Data.HPC_Data.NcMap', ...
    'eng.Data.HPT_Data.NcMap', 'eng.Data.LPT_Data.NcMap', ...
    'cntrl.Wfdmd', 'cntrl.N1cReq', 'cntrl.T45Lim', 'cntrl.WfqPs3Lim', 'cntrl.Ps3Lim'};
for c = {'FAN', 'LPC', 'HPC'}   % health-parameter read-back (proves the per-run values reached the blocks)
    for p = {'Nc', 'Wc', 'PR', 'Eff'}, s{end+1} = sprintf('eng.Data.%s_Data.s_C_%s', c{1}, p{1}); end %#ok<AGROW>
end
for c = {'HPT', 'LPT'}
    for p = {'Nc', 'Wc', 'PR', 'Eff'}, s{end+1} = sprintf('eng.Data.%s_Data.s_T_%s', c{1}, p{1}); end %#ok<AGROW>
end
end

function s = ssSignals()
s = {'converged', 'independents', 'Pa', 'S2.Pt', 'S21.Pt', 'S25.Pt', 'S36.Pt', 'S5.Pt', ...
    'S2.Tt', 'S25.Tt', 'S36.Tt', 'S45.Tt'};
end

function s = ssExtraSignals()
% phase B reference check point: margins and limit quantities of the converged SS point
% (fuel flow is independents[10] of the unchanged out_SS channels)
s = {'SM.SMFan', 'SM.SMLPC', 'SM.SMHPC', 'S4.Tt', 'S45.Tt', 'S48.Tt', 'S5.Tt', 'S36.Ps', ...
    'Shaft.N_Fan', 'Shaft.N_LPC', 'Shaft.N_HPC', 'Perf.Fnet', 'Perf.TSFC', ...
    'Data.FAN_Data.NcMap', 'Data.LPC_Data.NcMap', 'Data.HPC_Data.NcMap'};
end
