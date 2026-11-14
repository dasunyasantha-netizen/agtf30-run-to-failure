function [MWS, F] = cb_prepare(E, nSec, theta)
%CB_PREPARE  Package setup + flight inputs + steady-state ICs, like the campaign worker.
%   theta: struct array with fields mws_path ('MWS.HPT.s_Eff') and theta; the
%   carrier is set to healthy * (1 + theta) before define_inputs, so the
%   steady-state initial condition is solved with the degraded engine.
if nargin < 3, theta = struct('mws_path', {}, 'theta', {}); end
cd(E.engDir);
evalin('base', 'clear');
MWS = AGTF30.setup_simulation(struct('UseExcel', 0, 'LoadBus', 1, 'ICPoint', 'cruise'));
cd(E.engDir);
for i = 1:numel(theta)
    p = strsplit(erase(theta(i).mws_path, 'MWS.'), '.');
    h = getfield(MWS, p{:}); %#ok<GFLD>
    MWS = setfield(MWS, p{:}, h * (1 + theta(i).theta)); %#ok<SFLD>
end
M = readmatrix(E.csv, 'NumHeaderLines', 1);
M = M(1:nSec, :);
F.t = M(:, 1)'; F.alt = M(:, 2)'; F.mn = M(:, 3)'; F.dT = M(:, 4)'; F.pla = M(:, 5)';
MWS = define_inputs(MWS, struct('UseExcel', 0, 't', F.t, 'Alt', [F.alt; F.t], 'MN', [F.mn; F.t], ...
    'dT', [F.dT; F.t], 'PLA', [F.pla; F.t], 'ICPoint', 'auto'));
cd(E.engDir);
assignin('base', 'MWS', MWS);
end
