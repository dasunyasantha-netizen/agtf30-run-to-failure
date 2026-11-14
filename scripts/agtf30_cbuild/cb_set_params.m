function cb_set_params(MWS, F, NMAX)
%CB_SET_PARAMS  Put the CB_* Simulink.Parameter objects for one run into the base workspace.
%   MWS: package workspace after define_inputs (carriers applied, ICs solved);
%   F: flight (t, alt, mn, dT, pla rows). Used for Normal-mode runs of the
%   edited pkgc models (same values the C runner gets from its files).
if nargin < 3
    P0 = load(fullfile(fileparts(which('AGTF30SysDyn.mdl')), 'cb_params.mat'));
    NMAX = numel(P0.CB_In_t.Value);
end
V = struct();
for c = {'FAN', 'LPC', 'HPC', 'HPT', 'LPT'}
    for s = {'Nc', 'Wc', 'PR', 'Eff'}
        V.(sprintf('CB_%s_s_%s', c{1}, s{1})) = MWS.(c{1}).(['s_' s{1}]);
    end
end
V.CB_WfIC = MWS.In.WfIC; V.CB_VAFNIC = MWS.In.VAFNIC; V.CB_VBVIC = MWS.In.VBVIC;
V.CB_ICcom = MWS.In.ICcom; V.CB_N2IC = MWS.In.N2IC; V.CB_N3IC = MWS.In.N3IC;
for f = fieldnames(MWS.In.Sensor)'
    if startsWith(f{1}, 'IC'), V.(['CB_Sensor_' f{1}]) = MWS.In.Sensor.(f{1}); end
end
n = numel(F.t);
V.CB_In_t = [F.t, F.t(end) + (1:NMAX - n)];
V.CB_In_Alt = [F.alt, repmat(F.alt(end), 1, NMAX - n)];
V.CB_In_MN = [F.mn, repmat(F.mn(end), 1, NMAX - n)];
V.CB_In_dT = [F.dT, repmat(F.dT(end), 1, NMAX - n)];
V.CB_In_PLA = [F.pla, repmat(F.pla(end), 1, NMAX - n)];
V.CB_ssNc1 = MWS.In.ssNc1; V.CB_AltIC = MWS.In.AltIC; V.CB_MNIC = MWS.In.MNIC;
V.CB_dTambIC = MWS.In.dTambIC; V.CB_SSIC = [MWS.In.ICcom MWS.In.ICss]';
for f = fieldnames(V)'
    p = Simulink.Parameter(double(V.(f{1})));
    p.CoderInfo.StorageClass = 'ExportedGlobal';
    p.DataType = 'double';
    assignin('base', f{1}, p);
end
end
