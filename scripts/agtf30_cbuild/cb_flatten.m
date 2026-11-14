function S = cb_flatten(out)
%CB_FLATTEN  Every logged signal of a Simulink.SimulationOutput as flat columns.
%   S.names{i}, S.time{i} (column), S.data{i} (time x columns). Covers root
%   outputs (yout/tout), To Workspace variables (timeseries, structs of
%   timeseries, arrays) and Dataset logs (logsout), recursing into buses.
S = struct('names', {{}}, 'time', {{}}, 'data', {{}});
tout = [];
if any(strcmp(out.who, 'tout')), tout = out.get('tout'); end
for n = out.who'
    x = out.get(n{1});
    if strcmp(n{1}, 'tout'), S = add(S, 'tout', tout(:), tout(:)); continue; end
    S = walk(S, n{1}, x, tout);
end
end

function S = walk(S, name, x, tout)
if isa(x, 'Simulink.SimulationData.Dataset')
    for k = 1:x.numElements
        el = x.getElement(k);
        nm = el.Name;
        if isempty(nm), nm = sprintf('el%d', k); end
        if isprop(el, 'Values'), v = el.Values; else, v = el; end
        S = walk(S, [name '.' nm], v, tout);
    end
elseif isa(x, 'timeseries')
    d = x.Data;
    if ~x.IsTimeFirst
        nd = ndims(d);
        d = permute(d, [nd 1:nd-1]);
    end
    S = add(S, name, double(x.Time(:)), reshape(double(d), numel(x.Time), []));
elseif isstruct(x) && isfield(x, 'time') && isfield(x, 'signals')   % structure with time
    for k = 1:numel(x.signals)
        S = add(S, sprintf('%s.signals%d', name, k), double(x.time(:)), ...
            reshape(double(x.signals(k).values), numel(x.time), []));
    end
elseif isstruct(x)
    for f = fieldnames(x)'
        for k = 1:numel(x)
            nm = [name '.' f{1}];
            if numel(x) > 1, nm = sprintf('%s(%d).%s', name, k, f{1}); end
            S = walk(S, nm, x(k).(f{1}), tout);
        end
    end
elseif isnumeric(x) || islogical(x)
    if ~isempty(tout) && size(x, 1) == numel(tout)
        S = add(S, name, double(tout(:)), double(x));
    else
        S = add(S, name, nan(size(x, 1), 1), double(reshape(x, size(x, 1), [])));
    end
end
end

function S = add(S, name, t, d)
if any(strcmp(S.names, name))
    k = 2;
    while any(strcmp(S.names, sprintf('%s#%d', name, k))), k = k + 1; end
    name = sprintf('%s#%d', name, k);
end
S.names{end+1} = name; S.time{end+1} = t; S.data{end+1} = d;
end
