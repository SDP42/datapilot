"use client";

const MAX_VISIBLE_NODES = 8;
const LAYER_GAP = 150;
const NODE_RADIUS = 9;
const NODE_GAP = 28;
const TOP_PADDING = 40;
const SIDE_PADDING = 70;

interface Layer {
  label: string;
  size: number;
  kind: "endpoint" | "hidden";
}

/** A simple, dependency-free SVG diagram of an MLP's hidden-layer
 * structure — each hidden layer sized by its real, user-configured
 * neuron count, bracketed by generic input/output endpoint markers
 * (never a fabricated exact feature/class count, which this component
 * has no reliable way to know after one-hot encoding happens
 * server-side). A layer with more than `MAX_VISIBLE_NODES` neurons
 * shows a capped, representative column plus a "+N more" label rather
 * than drawing hundreds of circles. */
export function NetworkDiagram({ hiddenLayerSizes }: { hiddenLayerSizes: number[] }) {
  const layers: Layer[] = [
    { label: "Input", size: 1, kind: "endpoint" },
    ...hiddenLayerSizes.map((size, i) => ({
      label: `Hidden ${i + 1}`,
      size,
      kind: "hidden" as const,
    })),
    { label: "Output", size: 1, kind: "endpoint" },
  ];

  const maxVisible = Math.min(MAX_VISIBLE_NODES, Math.max(...layers.map((l) => l.size)));
  const height = TOP_PADDING * 2 + maxVisible * NODE_GAP;
  const width = SIDE_PADDING * 2 + (layers.length - 1) * LAYER_GAP;

  function nodeYPositions(size: number): number[] {
    const visible = Math.min(size, MAX_VISIBLE_NODES);
    const totalHeight = (visible - 1) * NODE_GAP;
    const startY = height / 2 - totalHeight / 2;
    return Array.from({ length: visible }, (_, i) => startY + i * NODE_GAP);
  }

  const layerX = (i: number) => SIDE_PADDING + i * LAYER_GAP;
  const colorForKind = (kind: Layer["kind"]) =>
    kind === "endpoint" ? "var(--accent)" : "var(--primary)";

  return (
    <div className="overflow-x-auto rounded-xl border border-surface-border bg-surface-2/30 p-4">
      <svg width={width} height={height} className="mx-auto">
        {layers.slice(0, -1).map((layer, i) => {
          const fromY = nodeYPositions(layer.size);
          const toY = nodeYPositions(layers[i + 1].size);
          const fromX = layerX(i);
          const toX = layerX(i + 1);
          return (
            <g key={`edges-${i}`} opacity={0.18}>
              {fromY.map((y1, a) =>
                toY.map((y2, b) => (
                  <line
                    key={`${a}-${b}`}
                    x1={fromX}
                    y1={y1}
                    x2={toX}
                    y2={y2}
                    stroke="var(--muted)"
                    strokeWidth={1}
                  />
                )),
              )}
            </g>
          );
        })}

        {layers.map((layer, i) => {
          const positions = nodeYPositions(layer.size);
          const x = layerX(i);
          const color = colorForKind(layer.kind);
          return (
            <g key={`${layer.label}-${i}`}>
              {positions.map((y, n) => (
                <circle
                  key={n}
                  cx={x}
                  cy={y}
                  r={NODE_RADIUS}
                  fill={color}
                  fillOpacity={0.85}
                  stroke={color}
                  strokeWidth={1.5}
                />
              ))}
              {layer.size > MAX_VISIBLE_NODES && (
                <text
                  x={x}
                  y={positions[positions.length - 1] + NODE_GAP}
                  textAnchor="middle"
                  fontSize={10}
                  fill="var(--muted)"
                >
                  +{layer.size - MAX_VISIBLE_NODES} more
                </text>
              )}
              <text
                x={x}
                y={height - TOP_PADDING + 24}
                textAnchor="middle"
                fontSize={12}
                fontWeight={600}
                fill="var(--foreground)"
              >
                {layer.label}
              </text>
              {layer.kind === "hidden" && (
                <text
                  x={x}
                  y={height - TOP_PADDING + 40}
                  textAnchor="middle"
                  fontSize={11}
                  fill="var(--muted)"
                >
                  {layer.size} unit{layer.size === 1 ? "" : "s"}
                </text>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
