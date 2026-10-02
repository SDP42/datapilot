function collectStyles(): string {
  let css = "";
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      const rules = sheet.cssRules;
      for (const rule of Array.from(rules)) css += rule.cssText + "\n";
    } catch {
      // cross-origin stylesheet (e.g. a font CDN) — can't read its rules, skip it
    }
  }
  return css;
}

function buildStandaloneDocument(container: HTMLElement, title: string, forPrint: boolean): string {
  const css = collectStyles();
  const printOverrides = forPrint
    ? `@page { margin: 16mm; } body { background: #07080f !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }`
    : "";
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>${title}</title>
<style>${css}\n${printOverrides}</style>
</head>
<body style="background:#07080f;color:#f4f5fb;padding:32px;margin:0;font-family:ui-sans-serif,system-ui,sans-serif;">
<h1 style="font-size:24px;font-weight:700;margin:0 0 4px;">${title}</h1>
<p style="color:#8b90a8;font-size:13px;margin:0 0 24px;">Exported from DataPilot — ${new Date().toLocaleString()}</p>
${container.outerHTML}
</body>
</html>`;
}

export function downloadDashboardHtml(container: HTMLElement, title: string) {
  const html = buildStandaloneDocument(container, title, false);
  const blob = new Blob([html], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${title.toLowerCase().replace(/\s+/g, "_")}.html`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function printDashboardPdf(container: HTMLElement, title: string) {
  const html = buildStandaloneDocument(container, title, true);
  const printWindow = window.open("", "_blank", "width=1100,height=800");
  if (!printWindow) return;
  printWindow.document.open();
  printWindow.document.write(html);
  printWindow.document.close();
  printWindow.onload = () => {
    printWindow.focus();
    printWindow.print();
  };
}
