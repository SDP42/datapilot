import type { CategoricalColumnAnalysis, HistogramBin, NumericDistribution } from "./api";

export type Row = Record<string, string>;

export function computeNumericDistribution(rows: Row[], column: string): NumericDistribution {
  const values = rows.map((r) => parseFloat(r[column])).filter((v) => Number.isFinite(v));
  const count = values.length;
  if (count === 0) {
    return {
      column,
      status: "unavailable",
      reason: "no numeric values in the current filter",
      count: 0,
      minimum: null,
      maximum: null,
      mean: null,
      median: null,
      std: null,
      histogram: { status: "unavailable", n_bins: null, bin_edges: [], bins: [], total_count: null },
    };
  }

  const min = Math.min(...values);
  const max = Math.max(...values);
  const mean = values.reduce((a, b) => a + b, 0) / count;
  const sorted = [...values].sort((a, b) => a - b);
  const median = sorted[Math.floor(count / 2)];
  const variance =
    count > 1 ? values.reduce((a, b) => a + (b - mean) ** 2, 0) / (count - 1) : 0;
  const std = Math.sqrt(variance);

  const nBins = Math.max(1, Math.min(12, Math.ceil(Math.log2(count)) + 1));
  let bins: HistogramBin[];
  if (max === min) {
    bins = [{ left_edge: min, right_edge: max, count }];
  } else {
    const width = (max - min) / nBins;
    const counts = new Array(nBins).fill(0);
    for (const v of values) {
      let idx = Math.floor((v - min) / width);
      if (idx >= nBins) idx = nBins - 1;
      if (idx < 0) idx = 0;
      counts[idx]++;
    }
    bins = counts.map((c, i) => ({
      left_edge: min + i * width,
      right_edge: min + (i + 1) * width,
      count: c,
    }));
  }

  return {
    column,
    status: "completed",
    count,
    minimum: min,
    maximum: max,
    mean,
    median,
    std,
    histogram: { status: "completed", n_bins: bins.length, bin_edges: [], bins, total_count: count },
  };
}

export function computeCategoricalAnalysis(
  rows: Row[],
  column: string,
  topN = 8,
): CategoricalColumnAnalysis {
  const counts = new Map<string, number>();
  let nonNull = 0;
  for (const r of rows) {
    const v = (r[column] ?? "").trim();
    if (!v) continue;
    nonNull++;
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const top = sorted
    .slice(0, topN)
    .map(([value, count]) => ({ value, count, frequency: nonNull ? count / nonNull : 0 }));

  return {
    column,
    count: nonNull,
    missing_count: rows.length - nonNull,
    missing_percentage: rows.length ? ((rows.length - nonNull) / rows.length) * 100 : 0,
    unique_count: counts.size,
    cardinality_ratio: rows.length ? counts.size / rows.length : null,
    top_values: top,
  };
}

export function distinctValues(rows: Row[], column: string): string[] {
  const set = new Set<string>();
  for (const r of rows) {
    const v = (r[column] ?? "").trim();
    if (v) set.add(v);
  }
  return [...set].sort((a, b) => a.localeCompare(b));
}

export function applyFilters(rows: Row[], filters: Record<string, string[]>): Row[] {
  const active = Object.entries(filters).filter(([, values]) => values.length > 0);
  if (active.length === 0) return rows;
  return rows.filter((r) => active.every(([col, values]) => values.includes(r[col])));
}

export interface NumericRange {
  min: number | null;
  max: number | null;
}

export interface DateRange {
  column: string;
  start: string | null;
  end: string | null;
}

export interface SearchFilter {
  columns: string[];
  query: string;
}

/** Combines every slicer type — categorical multi/single-select, numeric
 * range, date range, and free-text search — into one row predicate.
 * Each filter type is independent and AND-combined with the others. */
export function applyAllFilters(
  rows: Row[],
  opts: {
    categorical?: Record<string, string[]>;
    numericRanges?: Record<string, NumericRange>;
    dateRange?: DateRange | null;
    search?: SearchFilter | null;
  },
): Row[] {
  const categoricalActive = Object.entries(opts.categorical ?? {}).filter(
    ([, values]) => values.length > 0,
  );
  const rangeActive = Object.entries(opts.numericRanges ?? {}).filter(
    ([, r]) => r.min !== null || r.max !== null,
  );
  const dateActive = opts.dateRange && (opts.dateRange.start || opts.dateRange.end);
  const searchActive = opts.search && opts.search.query.trim().length > 0;

  if (!categoricalActive.length && !rangeActive.length && !dateActive && !searchActive) {
    return rows;
  }

  return rows.filter((r) => {
    for (const [col, values] of categoricalActive) {
      if (!values.includes(r[col])) return false;
    }
    for (const [col, range] of rangeActive) {
      const v = parseFloat(r[col]);
      if (!Number.isFinite(v)) return false;
      if (range.min !== null && v < range.min) return false;
      if (range.max !== null && v > range.max) return false;
    }
    if (dateActive && opts.dateRange) {
      const raw = r[opts.dateRange.column];
      const d = raw ? new Date(raw) : null;
      if (!d || Number.isNaN(d.getTime())) return false;
      if (opts.dateRange.start && d < new Date(opts.dateRange.start)) return false;
      if (opts.dateRange.end && d > new Date(opts.dateRange.end)) return false;
    }
    if (searchActive && opts.search) {
      const q = opts.search.query.trim().toLowerCase();
      const matches = opts.search.columns.some((c) => (r[c] ?? "").toLowerCase().includes(q));
      if (!matches) return false;
    }
    return true;
  });
}

export function numericBounds(rows: Row[], column: string): { min: number; max: number } | null {
  const values = rows.map((r) => parseFloat(r[column])).filter(Number.isFinite);
  if (values.length === 0) return null;
  return { min: Math.min(...values), max: Math.max(...values) };
}

function pearsonCorrelation(xs: number[], ys: number[]): number | null {
  const n = xs.length;
  if (n < 2) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i++) {
    const a = xs[i] - mx;
    const b = ys[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  if (dx === 0 || dy === 0) return null;
  return num / Math.sqrt(dx * dy);
}

export interface Insight {
  columnA: string;
  columnB: string;
  correlation: number;
  text: string;
}

/** The strongest pairwise Pearson correlation among the given numeric
 * columns in the current (filtered) rows — the "conclusion" row's
 * auto-generated insight. Returns null with fewer than 2 numeric columns
 * or no pair with enough paired, finite observations. */
export function strongestCorrelation(rows: Row[], numericCols: string[]): Insight | null {
  if (numericCols.length < 2) return null;
  let best: Insight | null = null;

  for (let i = 0; i < numericCols.length; i++) {
    for (let j = i + 1; j < numericCols.length; j++) {
      const a = numericCols[i];
      const b = numericCols[j];
      const xs: number[] = [];
      const ys: number[] = [];
      for (const r of rows) {
        const va = parseFloat(r[a]);
        const vb = parseFloat(r[b]);
        if (Number.isFinite(va) && Number.isFinite(vb)) {
          xs.push(va);
          ys.push(vb);
        }
      }
      if (xs.length < 5) continue;
      const corr = pearsonCorrelation(xs, ys);
      if (corr === null) continue;
      if (!best || Math.abs(corr) > Math.abs(best.correlation)) {
        const direction = corr > 0 ? "positively" : "negatively";
        const strength =
          Math.abs(corr) > 0.7 ? "strongly" : Math.abs(corr) > 0.4 ? "moderately" : "weakly";
        best = {
          columnA: a,
          columnB: b,
          correlation: corr,
          text:
            `${a} and ${b} are ${strength} ${direction} correlated (r = ${corr.toFixed(2)}) — ` +
            `as ${a} ${corr > 0 ? "increases" : "changes"}, ${b} tends to ` +
            `${corr > 0 ? "increase" : "decrease"} too.`,
        };
      }
    }
  }
  return best;
}

export interface TrendPoint {
  label: string;
  value: number;
}

/** Monthly trend of `valueCol`'s average (or a plain row count when
 * `valueCol` is null) bucketed by `dateCol`, from the current filtered
 * rows. Returns an empty array when no row has a parseable date. */
export function computeTrend(rows: Row[], dateCol: string, valueCol: string | null): TrendPoint[] {
  const buckets = new Map<string, { sum: number; count: number }>();
  for (const r of rows) {
    const raw = r[dateCol];
    if (!raw) continue;
    const d = new Date(raw);
    if (Number.isNaN(d.getTime())) continue;
    const label = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
    const entry = buckets.get(label) ?? { sum: 0, count: 0 };
    if (valueCol) {
      const v = parseFloat(r[valueCol]);
      if (Number.isFinite(v)) {
        entry.sum += v;
        entry.count += 1;
      }
    } else {
      entry.count += 1;
    }
    buckets.set(label, entry);
  }
  return [...buckets.entries()]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([label, e]) => ({ label, value: valueCol ? (e.count ? e.sum / e.count : 0) : e.count }));
}
