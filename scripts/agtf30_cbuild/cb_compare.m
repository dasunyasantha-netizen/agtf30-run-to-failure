function T = cb_compare(A, B, file)
%CB_COMPARE  Per-signal difference of B against reference A (both from cb_flatten).
%   max_abs            max |a - b| over samples and columns
%   max_rel            max_abs / max |a|  (relative to the signal's peak magnitude)
%   max_rel_pointwise  max over samples of |a - b| / max(|a|, 1e-6)
%   time_max_abs       max |t_a - t_b| (NaN if no time vector)
%   status  compared | missing_in_test | size_mismatch | missing_in_ref
n = numel(A.names);
signal = A.names(:); status = repmat({'compared'}, n, 1);
rowsA = nan(n, 1); rowsB = nan(n, 1); cols = nan(n, 1);
maxAbs = nan(n, 1); maxRel = nan(n, 1); maxRelPt = nan(n, 1); peak = nan(n, 1);
tAbs = nan(n, 1); nonfinite = zeros(n, 1);
for i = 1:n
    a = A.data{i}; rowsA(i) = size(a, 1); cols(i) = size(a, 2);
    j = find(strcmp(B.names, A.names{i}), 1);
    if isempty(j), status{i} = 'missing_in_test'; continue; end
    b = B.data{j}; rowsB(i) = size(b, 1);
    if ~isequal(size(a), size(b)), status{i} = 'size_mismatch'; continue; end
    ta = A.time{i}; tb = B.time{j};
    if numel(ta) == numel(tb) && any(isfinite(ta)), tAbs(i) = max(abs(ta - tb)); end
    fa = isfinite(a); fb = isfinite(b);
    nonfinite(i) = nnz(fa ~= fb);
    k = fa & fb;
    if ~any(k(:)), continue; end
    d = abs(a(k) - b(k));
    maxAbs(i) = max(d);
    peak(i) = max(abs(a(k)));
    if peak(i) > 0
        maxRel(i) = maxAbs(i) / peak(i);
    else
        maxRel(i) = 0; if maxAbs(i) > 0, maxRel(i) = inf; end
    end
    maxRelPt(i) = max(d ./ max(abs(a(k)), 1e-6));
end
T = table(signal, status, rowsA, rowsB, cols, peak, maxAbs, maxRel, maxRelPt, tAbs, nonfinite, ...
    'VariableNames', {'signal', 'status', 'rows_ref', 'rows_test', 'cols', 'ref_peak_abs', ...
    'max_abs', 'max_rel', 'max_rel_pointwise', 'time_max_abs', 'nonfinite_mismatch'});
extra = setdiff(B.names, A.names);
for i = 1:numel(extra)
    T = [T; {extra{i}, 'missing_in_ref', NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0}]; %#ok<AGROW>
end
if nargin > 2, writetable(T, file); end
end
