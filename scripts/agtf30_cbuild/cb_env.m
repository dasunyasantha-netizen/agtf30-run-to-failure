function E = cb_env(root, pkgName)
%CB_ENV  Private AGTF30 copy, T-MATS paths and Simulink cache for the C-build work.
%   The originals (AGTF30 v2.0 in AGTF30_SRC, T-MATS in TMATS_DIR) are only read; every model
%   change happens in a private copy: root\pkg (unchanged package, Normal/Rapid
%   references) or root\pkgc (code-generation copy edited by cb_make_models).
%   Only one copy is put on the path per MATLAB session (same model names).
%   Paths come from environment variables (see README); root defaults to AGTF30_WORK\agtf30_cbuild.
if nargin < 1 || isempty(root), root = fullfile(envor('AGTF30_WORK', 'work'), 'agtf30_cbuild'); end
if nargin < 2, pkgName = 'pkg'; end
E.root = root;
E.src = envor('AGTF30_SRC', 'AGTF30');                         % NASA AGTF30 v2.0 (original runs: commit a1521a76)
E.tmats = envor('TMATS_DIR', fullfile('T-MATS', 'Trunk'));     % NASA T-MATS v1.3.3, Trunk folder
E.repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
E.csv = fullfile(E.root, 'inputs', 'flight_u1c1.csv');   % copy of the speed-test flight
E.pkg = fullfile(root, pkgName);
if ~exist(E.pkg, 'dir')
    copyfile(E.src, E.pkg);
    if exist(fullfile(E.pkg, '.git'), 'dir'), rmdir(fullfile(E.pkg, '.git'), 's'); end
end
E.cache = fullfile(root, ['cache_' pkgName]);
if ~exist(E.cache, 'dir'), mkdir(E.cache); end
for d = {fullfile(E.tmats, 'TMATS_Library'), fullfile(E.tmats, 'TMATS_Library', 'MEX'), ...
        fullfile(E.tmats, 'TMATS_Library', 'TMATS_Support'), fullfile(E.tmats, 'TMATS_Tools'), ...
        fullfile(E.tmats, 'TMATS_Library', 'MATLAB_Scripts')}
    if exist(d{1}, 'dir'), addpath(d{1}); end
end
addpath(genpath(E.pkg));
addpath(fullfile(E.repo, 'scripts'));
addpath(fileparts(mfilename('fullpath')));
Simulink.fileGenControl('set', 'CacheFolder', E.cache, 'CodeGenFolder', E.cache, 'createDir', true);
E.engDir = fullfile(E.pkg, 'AGTF30');
E.model = 'AGTF30SysDyn';
warning('off', 'backtrace');
end

function v = envor(name, default)
%ENVOR  Value of environment variable NAME, or DEFAULT when it is not set.
v = getenv(name);
if isempty(v), v = default; end
end
