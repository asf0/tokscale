import type { UserEmbedStats } from "./getUserEmbedStats";

export type EmbedTheme = "dark" | "light";
export type EmbedSortBy = "tokens" | "cost";

export interface RenderProfileEmbedOptions {
  theme?: EmbedTheme;
  compact?: boolean;
  compactNumbers?: boolean;
  sortBy?: EmbedSortBy;
}

type ThemePalette = {
  background: string;
  panel: string;
  border: string;
  title: string;
  text: string;
  muted: string;
  accent: string;
};

const THEMES: Record<EmbedTheme, ThemePalette> = {
  dark: {
    background: "#0d1117",
    panel: "#161b22",
    border: "#30363d",
    title: "#ffffff",
    text: "#c9d1d9",
    muted: "#8b949e",
    accent: "#58a6ff",
  },
  light: {
    background: "#ffffff",
    panel: "#f6f8fa",
    border: "#d0d7de",
    title: "#1f2328",
    text: "#24292f",
    muted: "#57606a",
    accent: "#0969da",
  },
};

const FIGTREE_FONT_STACK = "Figtree, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
const FIGTREE_FONT_IMPORT = "https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700&amp;display=swap";

function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function formatCompact(value: number, kind: "number" | "currency"): string {
  const clamped = Math.max(0, value);

  if (kind === "currency") {
    const formatted = new Intl.NumberFormat("en-US", {
      notation: "compact",
      maximumFractionDigits: clamped >= 100 ? 1 : 2,
    }).format(clamped);
    return `$${formatted}`;
  }

  return new Intl.NumberFormat("en-US", {
    notation: "compact",
    maximumFractionDigits: clamped >= 100 ? 1 : 2,
  }).format(Math.round(clamped));
}

function formatNumber(value: number, compact = false): string {
  if (compact) return formatCompact(value, "number");
  return new Intl.NumberFormat("en-US").format(Math.max(0, Math.round(value)));
}

function formatCurrency(value: number, compact = false): string {
  if (compact) return formatCompact(value, "currency");
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(Math.max(0, value));
}

function formatDateLabel(value: string | null): string {
  if (!value) {
    return "No submissions yet";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Updated recently";
  }

  return `Updated ${new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(date)} (UTC)`;
}

// Approximate average character width for Figtree 15px semibold (weight 600).
// Used to estimate rendered username width for dynamic display-name positioning.
const APPROX_CHAR_WIDTH_15_SEMIBOLD = 9;

function getRankColor(rank: number | null, palette: ThemePalette): string {
  if (rank === 1) return "#EAB308";
  if (rank === 2) return "#9CA3AF";
  if (rank === 3) return "#D97706";
  return palette.text;
}

function metric(x: number, label: string, value: string, palette: ThemePalette, valueColor?: string): string {
  return [
    `<text x="${x}" y="112" fill="${palette.muted}" font-size="12" font-family="${FIGTREE_FONT_STACK}">${label}</text>`,
    `<text x="${x}" y="136" fill="${valueColor ?? palette.text}" font-size="20" font-weight="700" font-family="${FIGTREE_FONT_STACK}">${escapeXml(value)}</text>`,
  ].join("");
}

export function renderProfileEmbedSvg(
  data: UserEmbedStats,
  options: RenderProfileEmbedOptions = {}
): string {
  const theme: EmbedTheme = options.theme === "light" ? "light" : "dark";
  const compact = options.compact ?? false;
  const compactNumbers = options.compactNumbers ?? false;
  const sortBy: EmbedSortBy = options.sortBy === "cost" ? "cost" : "tokens";
  const palette = THEMES[theme];

  const width = compact ? 460 : 680;
  const height = compact ? 162 : 186;
  const rx = 12;

  const username = `@${data.user.username}`;
  const displayName = data.user.displayName ? escapeXml(data.user.displayName) : null;
  const tokens = formatNumber(data.stats.totalTokens, compactNumbers);
  const cost = formatCurrency(data.stats.totalCost, compactNumbers);
  const rank = data.stats.rank ? `#${data.stats.rank}` : "N/A";
  const updated = escapeXml(formatDateLabel(data.stats.updatedAt));
  const rankLabel = `Rank (${sortBy === "cost" ? "Cost" : "Tokens"})`;


  // Dynamically position the display name after the username to prevent overlap.
  // GitHub usernames can be up to 39 chars; at ~9px/char that's 360px which would
  // overlap a fixed x=140/156. We estimate the username pixel width and place the
  // display name after it with a small gap, hiding it if there's no room.
  const usernameEstimatedWidth = username.length * APPROX_CHAR_WIDTH_15_SEMIBOLD;
  const displayNameX = 24 + usernameEstimatedWidth + 8;
  const showDisplayName = displayName && displayNameX + 40 < width;

  const costColor = theme === "dark" ? "#3FB950" : "#1a7f37";
  const rankColor = getRankColor(data.stats.rank, palette);
  const metrics = [
    metric(compact ? 24 : 24, "Tokens", tokens, palette, palette.accent),
    metric(compact ? 184 : 194, "Cost", cost, palette, costColor),
    metric(compact ? 344 : 364, compact ? "Rank" : rankLabel, rank, palette, rankColor),
  ].join("");
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Tokscale profile stats for ${escapeXml(username)}">
  <defs>
    <style>@import url('${FIGTREE_FONT_IMPORT}');</style>
    <clipPath id="card-clip">
      <rect width="${width}" height="${height}" rx="${rx}"/>
    </clipPath>
  </defs>
  <rect width="${width}" height="${height}" rx="${rx}" fill="${palette.background}"/>
  <rect x="1" y="1" width="${width - 2}" height="${height - 2}" rx="${rx - 1}" fill="${palette.panel}" stroke="${palette.border}"/>
  <rect x="0" y="0" width="${width}" height="5" fill="${palette.accent}" clip-path="url(#card-clip)"/>
  <text x="24" y="36" fill="${palette.title}" font-size="18" font-weight="700" font-family="${FIGTREE_FONT_STACK}">Tokscale Stats</text>
  <text x="24" y="60" fill="${palette.text}" font-size="15" font-weight="600" font-family="${FIGTREE_FONT_STACK}">${escapeXml(username)}</text>
  ${
    showDisplayName
      ? `<text x="${displayNameX}" y="60" fill="${palette.muted}" font-size="13" font-family="${FIGTREE_FONT_STACK}">${displayName}</text>`
      : ""
  }

  ${metrics}
  <text x="24" y="${height - 16}" fill="${palette.muted}" font-size="11" font-family="${FIGTREE_FONT_STACK}">${updated}</text>
  <text x="${width - 158}" y="${height - 16}" fill="${palette.muted}" font-size="11" font-family="${FIGTREE_FONT_STACK}">tokscale.ai/u/${escapeXml(
    data.user.username
  )}</text>
</svg>`;
}

export function renderProfileEmbedErrorSvg(
  message: string,
  options: RenderProfileEmbedOptions = {}
): string {
  const theme: EmbedTheme = options.theme === "light" ? "light" : "dark";
  const palette = THEMES[theme];
  const safeMessage = escapeXml(message);

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg width="540" height="120" viewBox="0 0 540 120" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Tokscale embed error">
  <defs>
    <style>@import url('${FIGTREE_FONT_IMPORT}');</style>
    <clipPath id="card-clip">
      <rect width="540" height="120" rx="12"/>
    </clipPath>
  </defs>
  <rect width="540" height="120" rx="12" fill="${palette.background}"/>
  <rect x="1" y="1" width="538" height="118" rx="11" fill="${palette.panel}" stroke="${palette.border}"/>
  <rect x="0" y="0" width="540" height="5" fill="${palette.accent}" clip-path="url(#card-clip)"/>
  <text x="20" y="42" fill="${palette.title}" font-size="17" font-weight="700" font-family="${FIGTREE_FONT_STACK}">Tokscale Stats</text>
  <text x="20" y="72" fill="${palette.text}" font-size="13" font-family="${FIGTREE_FONT_STACK}">${safeMessage}</text>
  <text x="20" y="98" fill="${palette.muted}" font-size="11" font-family="${FIGTREE_FONT_STACK}">Try checking the username or submitting usage first.</text>
</svg>`;
}
