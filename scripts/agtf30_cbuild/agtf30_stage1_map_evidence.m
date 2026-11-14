function varargout = agtf30_stage1_map_evidence(action, varargin)
%AGTF30_STAGE1_MAP_EVIDENCE  Compressor-map observation routes and coverage (P1.S2).
%
%   agtf30_stage1_map_evidence('required')
%       The six required component/coordinate pairs: NcMap and Rline for
%       FAN, LPC and HPC.
%   agtf30_stage1_map_evidence('instrument', topModel)
%       On the loaded private copy, enables observation-only signal logging
%       (DataLogging, custom name) on the existing NcMap and Rline signals
%       inside the referenced engine model's FAN, LPC and HPC compressor
%       blocks, and confirms each one appears in the top model's logging set
%       (ModelLoggingInfo.createFromModel). Equations, control, solver
%       settings and installed libraries are untouched; nothing is saved.
%       Returns stable route metadata (block, port, logging name, channel).
%       P2.S4.M1: the package's steady-state IC generator (setup_InputsSS)
%       calls close_system('AGTF30_eng.mdl') after its sim, which discards
%       these in-memory marks; the replay therefore re-instruments
%       immediately before every transient.
%   agtf30_stage1_map_evidence('state', mapRoutes)
%       Non-mutating: whether the engine model is still loaded and each
%       established route still carries its DataLogging mark and name.
%   agtf30_stage1_map_evidence('persisted', ch, E)
%       The native logged map trajectories for the saved flight output
%       (shared time basis where identical), with their axis comparison.
%   agtf30_stage1_map_evidence('channels', mapRoutes)
%       The logged map channels named by the route metadata, read from the
%       simulation's signal-logging dataset (W_OUT or the base workspace).
%   agtf30_stage1_map_evidence('evidence', ch, specs, W, required, routes, span)
%       Coverage-aware excursion evidence: every required pair is observed or
%       an explicit gap; a total affected duration exists only when every
%       required pair is observed over the whole simulated span.
%   agtf30_stage1_map_evidence('check', name)
%       Non-simulating self-checks: 'rline_out_of_range', 'missing_rline',
%       'in_range_zero'. Returns [ok, detail].

switch action
    case 'required'
        varargout{1} = requiredPairs();
    case 'instrument'
        varargout{1} = instrument(varargin{:});
    case 'state'
        varargout{1} = routeState(varargin{:});
    case 'persisted'
        varargout{1} = persistedMap(varargin{:});
    case 'channels'
        varargout{1} = loggedChannels(varargin{:});
    case 'evidence'
        varargout{1} = evidence(varargin{:});
    case 'check'
        switch varargin{1}
            case 'rline_out_of_range', [varargout{1}, varargout{2}] = checkRlineOut();
            case 'missing_rline',      [varargout{1}, varargout{2}] = checkMissingRline();
            case 'in_range_zero',      [varargout{1}, varargout{2}] = checkInRangeZero();
            otherwise, error('agtf30s1:mapCheck', 'unknown map self-check %s', varargin{1});
        end
    otherwise
        error('agtf30s1:mapAction', 'unknown action %s', action);
end
end

% =========================================================================
% Required pairs and routes
% =========================================================================

function P = requiredPairs()
P = {};
comps = {'FAN', 'LPC', 'HPC'};
coords = {'NcMap', 'Rline'};
for i = 1:numel(comps)
    for j = 1:numel(coords)
        P{end+1} = struct('component', comps{i}, 'coordinate', coords{j}); %#ok<AGROW>
    end
end
end

function S = instrument(top)
%INSTRUMENT  Observation-only logging on the six existing map-coordinate
%   signals of the loaded private copy. Located by structure, not by name
%   guessing: each compressor is the T-MATS Compressor library block at its
%   engine-model path; Rline is the block's 'Rline' input (the solver's
%   R-line state fed into the map lookup); NcMap is the internal Demux output
%   whose line is named 'NcMap'.
locs = {'FAN', 'Engine/FrontEng/FAN'; 'LPC', 'Engine/FrontEng/LPC'; 'HPC', 'Engine/Core/HPC'};
S = struct('basis', ['DataLogging set in memory on the existing NcMap/Rline signals of the ' ...
        'private copy and confirmed in the top-model logging set; no sim, nothing saved'], ...
    'top_model', top, 'engine_model', '', 'model_block', '', 'signal_logging', '', ...
    'signal_logging_name', '', 'override_mode', '', 'model_block_mode', '', ...
    'routes', {{}}, 'required', 6, 'established_count', 0, 'established', false, ...
    'missing', {{}}, 'error', '');
req = requiredPairs();
for i = 1:numel(req)
    S.routes{i} = struct('component', req{i}.component, 'coordinate', req{i}.coordinate, ...
        'model', '', 'block', '', 'port', NaN, 'line_name', '', ...
        'logging_name', sprintf('agtf30s1_map_%s_%s', req{i}.component, req{i}.coordinate), ...
        'channel', '', 'data_logging_before', '', 'data_logging_set', false, ...
        'in_logging_set', false, 'established', false, 'reason', '');
end
try
    S.signal_logging = get_param(top, 'SignalLogging');
    S.signal_logging_name = get_param(top, 'SignalLoggingName');
    ov = get_param(top, 'DataLoggingOverride');
    S.override_mode = ov.LoggingMode;
    mb = find_system(top, 'LookUnderMasks', 'all', 'FollowLinks', 'on', ...
        'BlockType', 'ModelReference');
    mb = mb(cellfun(@(b) strcmp(get_param(b, 'ModelName'), 'AGTF30_eng'), mb));
    if numel(mb) ~= 1
        error('agtf30s1:engRef', 'expected one AGTF30_eng model block, found %d', numel(mb));
    end
    S.model_block = mb{1};
    S.model_block_mode = get_param(mb{1}, 'SimulationMode');
    eng = 'AGTF30_eng';
    load_system(eng);
    S.engine_model = get_param(eng, 'FileName');
catch err
    S.error = err.message;
    for i = 1:numel(S.routes), S.routes{i}.reason = ['engine model: ' err.message]; end
    S = finishRoutes(S);
    return
end
for i = 1:numel(S.routes)
    r = S.routes{i};
    r.model = 'AGTF30_eng';
    r.channel = [S.signal_logging_name '.' r.logging_name];
    try
        blk = ['AGTF30_eng/' locs{strcmp(locs(:, 1), r.component), 2}];
        if ~strcmp(get_param(blk, 'ReferenceBlock'), 'Lib_Turbo_Compressor_TMATS/Compressor')
            error('agtf30s1:mapBlock', '%s is not a T-MATS Compressor library block', blk);
        end
        if strcmp(r.coordinate, 'Rline')
            src = [blk '/Rline'];
            if ~strcmp(get_param(src, 'BlockType'), 'Inport')
                error('agtf30s1:mapRline', '%s is not an Inport', src);
            end
            ph = get_param(src, 'PortHandles');
            p = ph.Outport(1);
        else
            src = [blk '/Demux'];
            ph = get_param(src, 'PortHandles');
            names = arrayfun(@(h) lineName(h), ph.Outport, 'UniformOutput', false);
            k = find(strcmp(names, 'NcMap'));
            if numel(k) ~= 1
                error('agtf30s1:mapNcMap', '%s has %d outputs named NcMap', src, numel(k));
            end
            p = ph.Outport(k);
        end
        r.block = src;
        r.port = get_param(p, 'PortNumber');
        r.line_name = lineName(p);
        r.data_logging_before = get_param(p, 'DataLogging');
        set_param(p, 'DataLogging', 'on', 'DataLoggingNameMode', 'Custom', ...
            'DataLoggingName', r.logging_name);
        r.data_logging_set = strcmp(get_param(p, 'DataLogging'), 'on') && ...
            strcmp(get_param(p, 'DataLoggingName'), r.logging_name);
    catch err
        r.reason = err.message;
    end
    S.routes{i} = r;
end
% confirm against the top model's effective logging set
try
    if ~strcmp(S.signal_logging, 'on')
        error('agtf30s1:sigLog', 'top-model SignalLogging is %s', S.signal_logging);
    end
    mi = Simulink.SimulationData.ModelLoggingInfo.createFromModel(top);
    for i = 1:numel(S.routes)
        r = S.routes{i};
        if ~r.data_logging_set, continue; end
        for k = 1:numel(mi.Signals)
            s = mi.Signals(k);
            bp = s.BlockPath.convertToCell();
            if strcmp(bp{end}, r.block) && s.OutputPortIndex == r.port && ...
                    s.LoggingInfo.DataLogging && strcmp(s.LoggingInfo.LoggingName, r.logging_name)
                r.in_logging_set = true;
                break
            end
        end
        if ~r.in_logging_set
            r.reason = 'DataLogging set but the signal is absent from the top-model logging set';
        end
        S.routes{i} = r;
    end
catch err
    S.error = err.message;
    for i = 1:numel(S.routes)
        if isempty(S.routes{i}.reason)
            S.routes{i}.reason = ['logging set: ' err.message];
        end
    end
end
for i = 1:numel(S.routes)
    S.routes{i}.established = S.routes{i}.data_logging_set && S.routes{i}.in_logging_set;
end
try
    set_param('AGTF30_eng', 'Dirty', 'off');   % in-memory flag only; never saved
catch
end
S = finishRoutes(S);
end

function S = finishRoutes(S)
ok = cellfun(@(r) r.established, S.routes);
S.established_count = sum(ok);
S.established = all(ok) && numel(ok) == S.required;
S.missing = {};
for i = find(~ok)
    r = S.routes{i};
    S.missing{end+1} = sprintf('%s.%s: %s', r.component, r.coordinate, r.reason);
end
end

function n = lineName(ph)
n = '';
ln = get_param(ph, 'Line');
if ln ~= -1, n = get_param(ln, 'Name'); end
end

function T = routeState(M)
%ROUTESTATE  Whether the instrumented marks survived (no change is made).
T = struct('engine_loaded', false, 'routes_checked', 0, 'routes_marked', 0, ...
    'all_marked', false, 'detail', {{}}, 'error', '');
try
    T.engine_loaded = logical(bdIsLoaded('AGTF30_eng'));
catch err
    T.error = err.message;
    return
end
if ~isstruct(M) || ~isfield(M, 'routes'), return; end
for i = 1:numel(M.routes)
    r = M.routes{i};
    if ~r.established, continue; end
    T.routes_checked = T.routes_checked + 1;
    d = struct('pair', [r.component '.' r.coordinate], 'marked', false, 'reason', '');
    if ~T.engine_loaded
        d.reason = 'engine model not loaded (closed after instrumentation)';
    else
        try
            ph = get_param(r.block, 'PortHandles');
            p = ph.Outport(r.port);
            d.marked = strcmp(get_param(p, 'DataLogging'), 'on') && ...
                strcmp(get_param(p, 'DataLoggingName'), r.logging_name);
            if ~d.marked, d.reason = 'DataLogging mark absent on the loaded engine model'; end
        catch err
            d.reason = err.message;
        end
    end
    T.routes_marked = T.routes_marked + d.marked;
    T.detail{end+1} = d;
end
T.all_marked = T.routes_checked > 0 && T.routes_marked == T.routes_checked;
end

function P = persistedMap(ch, E)
%PERSISTEDMAP  Native logged map trajectories and their axis comparison for
%   the saved flight output; a missing or invalid coordinate is saved with
%   its status and no data (never zero-filled).
P = struct('basis', ['native logged NcMap/Rline samples, not resampled; time_s is shared ' ...
        'when every saved channel has the identical time vector'], ...
    'coverage_complete', E.coverage_complete, 'time_s', [], 'shared_time', false, ...
    'channels', struct('path', {}, 'component', {}, 'coordinate', {}, 'status', {}, ...
        'time_s', {}, 'data', {}, 'axis', {}, 'axis_range', {}, 'observed_range', {}, ...
        'seconds_outside', {}));
t0 = [];
shared = true;
for i = 1:numel(E.coverage)
    rec = E.coverage{i};
    t = []; v = [];
    j = [];
    if ~isempty(rec.channel)
        j = find(cellfun(@(x) strcmp(x.path, rec.channel), ch), 1);
    end
    if ~isempty(j) && any(strcmp(rec.status, {'observed', 'insufficient_coverage'}))
        t = ch{j}.time(:); v = ch{j}.data(:, 1);
        if isempty(t0)
            t0 = t;
        elseif ~isequal(t, t0)
            shared = false;
        end
    end
    P.channels(end+1) = struct('path', rec.channel, 'component', rec.component, ...
        'coordinate', rec.coordinate, 'status', rec.status, 'time_s', t, 'data', v, ...
        'axis', rec.axis, 'axis_range', rec.axis_range, ...
        'observed_range', rec.observed_range, 'seconds_outside', rec.seconds_outside);
end
if shared && ~isempty(t0)
    P.shared_time = true;
    P.time_s = t0;
    for i = 1:numel(P.channels)
        if ~isempty(P.channels(i).time_s), P.channels(i).time_s = []; end
    end
end
end

function ch = loggedChannels(M)
%LOGGEDCHANNELS  Route-named map channels from the signal-logging dataset.
ch = {};
if ~isstruct(M) || ~isfield(M, 'routes') || isempty(M.signal_logging_name), return; end
name = M.signal_logging_name;
ds = [];
if evalin('base', 'exist(''W_OUT'', ''var'')') == 1
    w = evalin('base', 'W_OUT');
    if isa(w, 'Simulink.SimulationOutput') && any(strcmp(w.who, name))
        ds = w.get(name);
    end
end
if isempty(ds) && evalin('base', sprintf('exist(''%s'', ''var'')', name)) == 1
    ds = evalin('base', name);
end
if ~isa(ds, 'Simulink.SimulationData.Dataset'), return; end
for i = 1:numel(M.routes)
    r = M.routes{i};
    if ~r.established, continue; end
    el = ds.getElement(r.logging_name);
    if isa(el, 'Simulink.SimulationData.Dataset') || isempty(el)
        continue        % absent or ambiguous: left as an explicit coverage gap
    end
    ts = el.Values;
    t = double(ts.Time(:));
    v = double(squeeze(ts.Data));
    v = v(:);
    first = NaN; last = NaN;
    if ~isempty(t), first = t(1); last = t(end); end
    ch{end+1} = struct('path', r.channel, 'samples', numel(t), 'first_time', first, ...
        'last_time', last, 'finite', all(isfinite(v)), 'time', t, 'data', v); %#ok<AGROW>
end
end

% =========================================================================
% Coverage-aware excursion evidence
% =========================================================================

function E = evidence(ch, specs, W, required, routes, span)
tol = 0.02;
isMap = @(w) startsWith(w.source, 'tmats_native') && ...
    ~isempty(regexpi(w.message, 'Map|SPR|SMWc|SMPR|SM solver', 'once'));
nat = W(cellfun(isMap, W));
E = struct('native_messages', {cellfun(@(w) struct('block', w.block, 'message', w.message), ...
        nat, 'UniformOutput', false)}, ...
    'native_message_count', numel(nat), 'native_messages_latched', true, ...
    'native_note', ['T-MATS prints each map warning once per block (IWork latch) and ' ...
        'states no time; the count is not a duration'], ...
    'span_s', span, 'coverage_tolerance_s', tol, ...
    'required_pairs', {cellfun(@(p) [p.component '.' p.coordinate], required, ...
        'UniformOutput', false)}, ...
    'observed_pairs', {{}}, 'missing_pairs', {{}}, 'coverage', {{}}, ...
    'coverage_complete', false, 'timing_claim', 'none', 'timed_evidence_available', false, ...
    'affected_seconds_total', NaN, 'affected_intervals', {{}}, ...
    'observed_scope_affected_seconds', NaN, ...
    'observed_scope_note', ['union of outside-axis time over the observed coordinates only; ' ...
        'a lower bound when coverage is incomplete, never the total'], ...
    'per_coordinate_note', ['seconds_outside is per coordinate; coordinates overlap in ' ...
        'time, so they are never summed into flight time'], ...
    'basis', ['route-named logged NcMap/Rline channels against the map axis ranges; an ' ...
        'interval runs from its first outside sample to the next native sample (or the ' ...
        'last recorded sample)'], 'gap', '');
allIv = zeros(0, 2);
for i = 1:numel(required)
    c = required{i}.component; k = required{i}.coordinate;
    rec = struct('component', c, 'coordinate', k, 'status', '', 'reason', '', ...
        'channel', '', 'axis', '', 'axis_range', [NaN NaN], 'recorded_span', [NaN NaN], ...
        'intervals', {{}}, 'seconds_outside', NaN, 'samples_outside', NaN, ...
        'samples', 0, 'observed_range', [NaN NaN], 'margin_above_axis_min', NaN, ...
        'margin_below_axis_max', NaN);
    sp = pick(specs, c, k);
    r = pick(routes, c, k);
    if isempty(sp)
        rec.status = 'missing_axis';
        rec.reason = 'no map axis range for this coordinate in the setup MWS';
    elseif isempty(r) || ~r.established
        rec.status = 'missing_route';
        rec.reason = 'no established logging route';
        if ~isempty(r) && ~isempty(r.reason), rec.reason = r.reason; end
    else
        rec.channel = r.channel;
        rec.axis = sp.axis; rec.axis_range = [sp.lo sp.hi];
        j = find(cellfun(@(x) strcmp(x.path, r.channel), ch), 1);
        if isempty(j)
            rec.status = 'missing_channel';
            rec.reason = 'route established but the channel is absent from the output';
        else
            t = ch{j}.time(:); v = ch{j}.data(:, 1);
            if ~isempty(t), rec.recorded_span = [t(1) t(end)]; end
            rec.samples = numel(t);
            if ~isempty(v) && all(isfinite(v))
                % axis comparison over the recorded samples (negative margin = outside)
                rec.observed_range = [min(v) max(v)];
                rec.margin_above_axis_min = min(v) - sp.lo;
                rec.margin_below_axis_max = sp.hi - max(v);
            end
            if numel(t) < 2 || ~all(isfinite(t)) || ~all(isfinite(v)) || any(diff(t) < 0)
                rec.status = 'invalid_trajectory';
                rec.reason = 'fewer than 2 samples, non-finite values or non-monotonic time';
            elseif numel(span) ~= 2 || ~all(isfinite(span)) || span(2) <= span(1)
                rec.status = 'insufficient_coverage';
                rec.reason = 'no simulated span to cover';
            elseif t(1) > span(1) + tol || t(end) < span(2) - tol
                rec.status = 'insufficient_coverage';
                rec.reason = sprintf('recorded [%g, %g] s does not cover simulated [%g, %g] s', ...
                    t(1), t(end), span(1), span(2));
            else
                rec.status = 'observed';
                rec.intervals = outsideIntervals(t, v, sp.lo, sp.hi);
                rec.seconds_outside = sum(cellfun(@(x) x.duration_s, rec.intervals));
                rec.samples_outside = sum(cellfun(@(x) x.samples, rec.intervals));
                for q = 1:numel(rec.intervals)
                    allIv(end+1, :) = [rec.intervals{q}.start_s rec.intervals{q}.end_s]; %#ok<AGROW>
                end
            end
        end
    end
    E.coverage{end+1} = rec;
    if strcmp(rec.status, 'observed')
        E.observed_pairs{end+1} = [c '.' k];
    else
        E.missing_pairs{end+1} = sprintf('%s.%s (%s)', c, k, rec.status);
    end
end
[U, secs] = unionIntervals(allIv);
if ~isempty(E.observed_pairs)
    E.observed_scope_affected_seconds = secs;
end
E.coverage_complete = ~isempty(required) && isempty(E.missing_pairs);
E.timed_evidence_available = E.coverage_complete;
if E.coverage_complete
    E.timing_claim = 'complete';
    E.affected_seconds_total = secs;
    E.affected_intervals = arrayfun(@(q) struct('start_s', U(q, 1), 'end_s', U(q, 2), ...
        'duration_s', U(q, 2) - U(q, 1)), (1:size(U, 1))', 'UniformOutput', false)';
else
    if ~isempty(E.observed_pairs), E.timing_claim = 'partial'; end
    E.gap = ['map-excursion timing incomplete; total affected time unknown. Missing: ' ...
        strjoin(E.missing_pairs, '; ')];
end
end

function x = pick(list, c, k)
x = [];
for i = 1:numel(list)
    if strcmp(list{i}.component, c) && strcmp(list{i}.coordinate, k)
        x = list{i};
        return
    end
end
end

function [U, secs] = unionIntervals(iv)
U = zeros(0, 2);
if ~isempty(iv)
    iv = sortrows(iv, 1);
    U = iv(1, :);
    for q = 2:size(iv, 1)
        if iv(q, 1) <= U(end, 2)
            U(end, 2) = max(U(end, 2), iv(q, 2));
        else
            U(end+1, :) = iv(q, :); %#ok<AGROW>
        end
    end
end
secs = sum(U(:, 2) - U(:, 1));
end

function iv = outsideIntervals(t, v, lo, hi)
out = v < lo | v > hi;
iv = {};
n = numel(t);
k = 1;
while k <= n
    if ~out(k)
        k = k + 1;
        continue
    end
    j = k;
    while j < n && out(j + 1), j = j + 1; end
    if j < n, e = t(j + 1); else, e = t(j); end
    seg = v(k:j);
    iv{end+1} = struct('start_s', t(k), 'end_s', e, 'duration_s', e - t(k), ...
        'samples', j - k + 1, 'min_value', min(seg), 'max_value', max(seg)); %#ok<AGROW>
    k = j + 1;
end
end

% =========================================================================
% Self-checks (synthetic data only; no model, no simulation)
% =========================================================================

function [ch, specs, routes, req] = fanCase(ncmap, rline, tt)
req = {struct('component', 'FAN', 'coordinate', 'NcMap'), ...
       struct('component', 'FAN', 'coordinate', 'Rline')};
specs = {struct('component', 'FAN', 'coordinate', 'NcMap', 'axis', 'selfcheck.NcVec', ...
             'lo', 0, 'hi', 2), ...
         struct('component', 'FAN', 'coordinate', 'Rline', 'axis', 'selfcheck.RVec', ...
             'lo', 1, 'hi', 2.5)};
routes = {route('NcMap'), route('Rline')};
ch = {};
if ~isempty(ncmap), ch{end+1} = chan('logsout.agtf30s1_map_FAN_NcMap', tt, ncmap); end
if ~isempty(rline), ch{end+1} = chan('logsout.agtf30s1_map_FAN_Rline', tt, rline); end
end

function r = route(k)
r = struct('component', 'FAN', 'coordinate', k, 'established', true, 'reason', '', ...
    'channel', ['logsout.agtf30s1_map_FAN_' k]);
end

function c = chan(path, t, v)
c = struct('path', path, 'samples', numel(t), 'first_time', t(1), 'last_time', t(end), ...
    'finite', all(isfinite(v)), 'time', t(:), 'data', v(:));
end

function [ok, D] = checkRlineOut()
%CHECKRLINEOUT  NcMap in range throughout, Rline above its axis for [4, 6):
%   the Rline interval is known and the total affected time is 2 s.
tt = (0:0.5:10)';
nc = ones(size(tt));
rl = 1.5 * ones(size(tt));
rl(tt >= 4 & tt < 6) = 3;
[ch, specs, routes, req] = fanCase(nc, rl, tt);
E = evidence(ch, specs, {}, req, routes, [0 10]);
D = struct('evidence', E, 'expected_rline', [4 6 2], 'expected_total_s', 2);
iv = E.coverage{2}.intervals;
ok = E.coverage_complete && strcmp(E.timing_claim, 'complete') && numel(iv) == 1 && ...
    isequal([iv{1}.start_s iv{1}.end_s iv{1}.duration_s], D.expected_rline) && ...
    E.coverage{1}.seconds_outside == 0 && isempty(E.coverage{1}.intervals) && ...
    E.affected_seconds_total == 2 && isempty(E.missing_pairs);
end

function [ok, D] = checkMissingRline()
%CHECKMISSINGRLINE  A missing, unrouted, short or invalid Rline must leave an
%   explicit gap, no complete timing claim and an unknown (null) total, even
%   though the observed NcMap is in range.
tt = (0:0.5:10)';
nc = ones(size(tt));
cases = {};
[ch, specs, routes, req] = fanCase(nc, [], tt);                 % channel absent
cases{end+1} = {'missing_channel', ch, routes};
r2 = routes; r2{2}.established = false; r2{2}.reason = 'selfcheck: route not found';
cases{end+1} = {'missing_route', ch, r2};
short = chan('logsout.agtf30s1_map_FAN_Rline', tt(tt <= 5), 1.5 * ones(sum(tt <= 5), 1));
cases{end+1} = {'insufficient_coverage', [ch, {short}], routes};
bad = 1.5 * ones(size(tt)); bad(3) = NaN;
cases{end+1} = {'invalid_trajectory', [ch, {chan('logsout.agtf30s1_map_FAN_Rline', tt, bad)}], ...
    routes};
D = struct('cases', {{}});
ok = true;
for i = 1:numel(cases)
    E = evidence(cases{i}{2}, specs, {}, req, cases{i}{3}, [0 10]);
    js = jsonencode(E);
    rec = struct('expected_status', cases{i}{1}, 'status', E.coverage{2}.status, ...
        'coverage_complete', E.coverage_complete, 'timing_claim', E.timing_claim, ...
        'missing_pairs', {E.missing_pairs}, 'serialized_total_is_null', ...
        contains(js, '"affected_seconds_total":null'), ...
        'observed_scope_affected_seconds', E.observed_scope_affected_seconds);
    D.cases{end+1} = rec;
    ok = ok && strcmp(rec.status, rec.expected_status) && ~E.coverage_complete && ...
        ~E.timed_evidence_available && strcmp(E.timing_claim, 'partial') && ...
        isnan(E.affected_seconds_total) && rec.serialized_total_is_null && ...
        isempty(E.affected_intervals) && strcmp(E.coverage{1}.status, 'observed') && ...
        E.observed_scope_affected_seconds == 0 && ~isempty(E.gap);
end
end

function [ok, D] = checkInRangeZero()
%CHECKINRANGEZERO  Both coordinates observed and in range over the whole
%   span: a legitimate, complete zero.
tt = (0:0.5:10)';
[ch, specs, routes, req] = fanCase(ones(size(tt)), 1.5 * ones(size(tt)), tt);
E = evidence(ch, specs, {}, req, routes, [0 10]);
D = struct('evidence', E);
ok = E.coverage_complete && strcmp(E.timing_claim, 'complete') && ...
    E.affected_seconds_total == 0 && isempty(E.missing_pairs) && ...
    contains(jsonencode(E), '"affected_seconds_total":0');
end
