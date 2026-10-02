/** A small, dependency-free CSV parser (quoted fields, embedded commas/
 * newlines, escaped quotes) — used client-side so dashboard slicers can
 * filter the actual rows instead of only the backend's static aggregates. */
export function parseCsv(text: string): { headers: string[]; rows: Record<string, string>[] } {
  const table: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += c;
      }
      continue;
    }
    if (c === '"') {
      inQuotes = true;
    } else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      field = "";
      if (row.length > 1 || row[0] !== "") table.push(row);
      row = [];
    } else {
      field += c;
    }
  }
  if (field !== "" || row.length > 0) {
    row.push(field);
    table.push(row);
  }
  if (table.length === 0) return { headers: [], rows: [] };

  const headers = table[0];
  const rows = table.slice(1).map((r) => {
    const obj: Record<string, string> = {};
    headers.forEach((h, i) => {
      obj[h] = r[i] ?? "";
    });
    return obj;
  });
  return { headers, rows };
}

export async function parseCsvFile(
  file: File,
): Promise<{ headers: string[]; rows: Record<string, string>[] }> {
  const text = await file.text();
  return parseCsv(text);
}
