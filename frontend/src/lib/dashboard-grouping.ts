/**
 * Groups a flat list of chartable columns into N domain-themed dashboards.
 *
 * Categorization is a deterministic keyword match against common analytics
 * domains (HR, supply chain, sales/finance, operations, …) — no AI call,
 * so it is instant and reproducible. When the detected category count
 * doesn't match the requested dashboard count, the smallest groups are
 * merged or the largest group is split until it does.
 */

interface CategoryDef {
  name: string;
  keywords: string[];
}

const CATEGORY_DEFS: CategoryDef[] = [
  {
    name: "Demographics & Workforce",
    keywords: [
      "age", "gender", "employee", "tenure", "department", "hire", "headcount",
      "staff", "workforce", "education", "marital", "ethnicity", "role", "title",
    ],
  },
  {
    name: "Compensation & Performance",
    keywords: [
      "salary", "pay", "compensation", "bonus", "rating", "performance",
      "promotion", "review", "income", "wage",
    ],
  },
  {
    name: "Attrition & Engagement",
    keywords: [
      "attrition", "turnover", "resign", "exit", "satisfaction", "engagement",
      "absenteeism", "leave", "churn",
    ],
  },
  {
    name: "Inventory & Stock",
    keywords: ["inventory", "stock", "warehouse", "sku", "quantity", "reorder", "backorder"],
  },
  {
    name: "Logistics & Shipping",
    keywords: [
      "shipment", "delivery", "transit", "carrier", "route", "logistics",
      "freight", "leadtime", "lead_time", "dispatch", "transport",
    ],
  },
  {
    name: "Suppliers & Procurement",
    keywords: ["supplier", "vendor", "procurement", "purchase", "po_number", "sourcing"],
  },
  {
    name: "Sales & Revenue",
    keywords: [
      "sales", "revenue", "price", "profit", "margin", "discount", "order",
      "customer", "transaction", "deal",
    ],
  },
  {
    name: "Operations & Quality",
    keywords: [
      "defect", "quality", "downtime", "efficiency", "utilization", "production",
      "yield", "throughput", "capacity",
    ],
  },
  {
    name: "Finance & Cost",
    keywords: ["cost", "expense", "budget", "invoice", "payment", "tax", "spend"],
  },
  {
    name: "Time & Dates",
    keywords: ["date", "timestamp", "year", "month", "quarter", "fiscal"],
  },
];

export function categorizeColumn(column: string): string {
  const lower = column.toLowerCase();
  for (const def of CATEGORY_DEFS) {
    if (def.keywords.some((k) => lower.includes(k))) return def.name;
  }
  return "General";
}

export interface DashboardGroup {
  title: string;
  columns: string[];
}

export function groupColumnsIntoDashboards(
  columns: string[],
  dashboardCount: number,
): DashboardGroup[] {
  const target = Math.max(1, Math.min(dashboardCount, columns.length || 1));

  const byCategory = new Map<string, string[]>();
  for (const c of columns) {
    const cat = categorizeColumn(c);
    if (!byCategory.has(cat)) byCategory.set(cat, []);
    byCategory.get(cat)!.push(c);
  }

  const groups: DashboardGroup[] = Array.from(byCategory.entries()).map(([title, cols]) => ({
    title,
    columns: cols,
  }));

  // Merge the two smallest groups until we're at or below the target count.
  while (groups.length > target) {
    groups.sort((a, b) => a.columns.length - b.columns.length);
    const a = groups.shift()!;
    const b = groups.shift()!;
    groups.push({
      title: a.title === b.title ? a.title : `${a.title} & ${b.title}`,
      columns: [...a.columns, ...b.columns],
    });
  }

  // Split the largest group until we reach the target count (or can't split further).
  while (groups.length < target) {
    groups.sort((a, b) => b.columns.length - a.columns.length);
    const largest = groups[0];
    if (!largest || largest.columns.length <= 1) break;
    const mid = Math.ceil(largest.columns.length / 2);
    groups[0] = { title: `${largest.title} (1)`, columns: largest.columns.slice(0, mid) };
    groups.push({ title: `${largest.title} (2)`, columns: largest.columns.slice(mid) });
  }

  groups.sort((a, b) => b.columns.length - a.columns.length);
  return groups;
}

/** A reasonable default dashboard count: the number of distinct detected
 * categories among the given columns, clamped to a sane range. */
export function suggestDashboardCount(columns: string[]): number {
  const categories = new Set(columns.map(categorizeColumn));
  return Math.max(1, Math.min(categories.size, columns.length || 1));
}
