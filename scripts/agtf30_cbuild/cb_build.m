function cb_build(pkgName, genName, onlySS, Ts)
%CB_BUILD  Generate GRT C code for the edited AGTF30SysDyn / AGTF30SysSS (pkgc) and package it.
%   1. cb_make_models once (edits saved in %AGTF30_WORK%\agtf30_cbuild\pkgc only);
%   2. CB_* parameters from the A1 healthy MWS (original package, same flight);
%   3. root-output widths -> expanded column names (outputs_<model>.txt);
%   4. slbuild (code only) and packNGo (flat zip) per model, plus sources.txt
%      (the C files the build needs, without GRT's own main);
%   5. ic_tables.json: everything the Python IC generator needs (setup_InputsSS
%      'auto' branch: SS_Mtrx tables, controller schedules, solver constants).
%   Output folder: %AGTF30_WORK%\agtf30_cbuild\gen\<model>\.
%   Phase B variant: cb_build('pkgs', 'gen_ss', true) builds only AGTF30SysSS with the
%   extra reference-point channels (cb_make_models ssExtra) in a separate copy/folder.
%   Phase C variant: cb_build('pkgf', 'gen_fine', false, 0.0075) builds both models with the
%   fixed step MWS.In.Ts halved (decision #7 retry 'smaller step'); MWS.In.Ts is the model's
%   FixedStep and every discrete block's sample time, so all rates change consistently.
if nargin < 1 || isempty(pkgName), pkgName = 'pkgc'; end
if nargin < 2 || isempty(genName), genName = 'gen'; end
if nargin < 3, onlySS = false; end
if nargin < 4, Ts = []; end
E = cb_env([], pkgName);
if ~exist(fullfile(E.engDir, 'cb_models.json'), 'file')
    cb_make_models([], pkgName, onlySS);
end
genRoot = fullfile(E.root, genName); if ~exist(genRoot, 'dir'), mkdir(genRoot); end
L = load(fullfile(E.root, 'a1', 'mws_healthy.mat'));
MWS = L.MWS;
if ~isempty(Ts), MWS.In.Ts = Ts; fprintf('cb_build: MWS.In.Ts = %g\n', Ts); end
assignin('base', 'MWS', MWS);
cd(E.engDir);
evalin('base', 'load(''SimSetup/Eng_Bus.mat'')');
cb_set_params(MWS, L.F);
info = jsondecode(fileread(fullfile(E.engDir, 'cb_models.json')));
models = {'AGTF30SysDyn', info.dyn_outputs; 'AGTF30SysSS', info.ss_outputs};
if onlySS, models = models(2, :); end
for k = 1:size(models, 1)
    m = models{k, 1};
    out = fullfile(genRoot, m); if ~exist(out, 'dir'), mkdir(out); end
    load_system(m);
    names = expandedNames(m, models{k, 2});
    writelines(names, fullfile(out, ['outputs_' m '.txt']));
    slbuild(m);
    bi = load(fullfile(E.cache, [m '_grt_rtw'], 'buildInfo.mat'));
    zipFile = fullfile(out, [m '_cb.zip']);
    packNGo(bi.buildInfo, 'packType', 'flat', 'fileName', zipFile);
    src = getSourceFiles(bi.buildInfo, false, false);
    writelines(src(:), fullfile(out, 'sources.txt'));
    if strcmp(m, 'AGTF30SysDyn')
        % Transport Delay 'Wf delay' (fuel actuator): its initial output MWS.In.WfIC -> CB_WfIC
        % is not tunable in generated code and appears as a literal; cb_compile.py puts the
        % global back (exactly 2 sites: buffer start value and initial output)
        P = struct('file', 'AGTF30SysDyn.c', 'find', sprintf('(%.17g)', MWS.In.WfIC), ...
            'replace', '(CB_WfIC)', 'count', 2);
        fid = fopen(fullfile(out, 'patches.json'), 'w'); fprintf(fid, '%s', jsonencode({P})); fclose(fid);
    end
    if strcmp(m, 'AGTF30SysSS')   % discrete model without time in the generated code
        writelines({sprintf('CB_FIXED_STEP=%.17g', evalin('base', get_param(m, 'FixedStep'))); ...
            sprintf('CB_DEFAULT_STOP=%.17g', MWS.In.Max_IterSS)}, fullfile(out, 'defines.txt'));
    else
        writelines({''}, fullfile(out, 'defines.txt'));
    end
    fprintf('built %s: %d outputs, zip %s\n', m, numel(names), zipFile);
end
writeIcTables(MWS, fullfile(genRoot, 'ic_tables.json'));
fprintf('CB_BUILD_DONE\n');
end

function names = expandedNames(m, sigs)
feval(m, [], [], [], 'compile');
c = onCleanup(@() feval(m, [], [], [], 'term'));
ph = get_param([m '/cb_sel'], 'PortHandles');
op = ph.Outport;
if getSimulinkBlockHandle([m '/cb_sel2']) > 0   % phase B SS variant: second selector
    ph2 = get_param([m '/cb_sel2'], 'PortHandles');
    op = [op, ph2.Outport];
end
if numel(op) ~= numel(sigs), error('cb:sel', '%s: %d selector ports for %d signals', m, numel(op), numel(sigs)); end
names = {};
for i = 1:numel(sigs)
    w = get_param(op(i), 'CompiledPortWidth');
    if w == 1
        names{end+1} = sigs{i}; %#ok<AGROW>
    else
        for j = 1:w, names{end+1} = sprintf('%s[%d]', sigs{i}, j); end %#ok<AGROW>
    end
end
names = names(:);
end

function writeIcTables(MWS, file)
S = load(fullfile(MWS.top_level, 'SimSetup', 'SS_Mtrx.mat'));
SS = S.SS;
T = struct('Alt', SS.Alt(:)', 'Nc', SS.Nc(:)', 'MN', SS.MN(:)', 'Ncmin', SS.Ncmin, 'Ncmax', SS.Ncmax);
T.Ind = cell(1, 12);
for i = 1:12, T.Ind{i} = SS.Ind{i}{1}; end
T.Ind_size = size(SS.Ind{1}{1});
C = MWS.Cntrl;
T.PM_AltVec = C.PM_AltVec; T.PM_TF = C.PM_TF; T.PM_MNVec = C.PM_MNVec; T.PM_NcArray = C.PM_NcArray;
T.PM_NcArray_size = size(C.PM_NcArray);
T.VAFN_MN = C.VAFN_MN; T.VAFN_Nc1 = C.VAFN_Nc1; T.VAFN_sch = C.VAFN_sch;
T.VBV_MN = C.VBV_MN; T.VBV_Nc1 = C.VBV_Nc1; T.VBV_sch = C.VBV_sch;
T.FAN_s_Nc = MWS.FAN.s_Nc;
T.manual = struct('WfManEn', MWS.In.WfManEn, 'N1ManEn', MWS.In.N1ManEn, ...
    'VAFNManEn', MWS.In.VAFNManEn, 'VBVManEn', MWS.In.VBVManEn);
T.note = ['Arrays are MATLAB column-major; *_size gives the MATLAB size. ' ...
    'Exported by scripts/agtf30_cbuild/cb_build.m from the AGTF30 package.'];
fid = fopen(file, 'w'); fprintf(fid, '%s', jsonencode(T, 'ConvertInfAndNaN', true)); fclose(fid);
end
