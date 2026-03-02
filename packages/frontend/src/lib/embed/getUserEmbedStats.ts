import { unstable_cache } from "next/cache";
import { db, users, submissions } from "@/lib/db";
import { eq, sql } from "drizzle-orm";

export type EmbedSortBy = "tokens" | "cost";

export interface UserEmbedStats {
  user: {
    id: string;
    username: string;
    displayName: string | null;
    avatarUrl: string | null;
  };
  stats: {
    totalTokens: number;
    totalCost: number;
    submissionCount: number;
    rank: number | null;
    updatedAt: string | null;
  };
}

async function fetchUserEmbedStats(username: string, sortBy: EmbedSortBy): Promise<UserEmbedStats | null> {
  const [result] = await db
    .select({
      id: users.id,
      username: users.username,
      displayName: users.displayName,
      avatarUrl: users.avatarUrl,
      totalTokens: sql<number>`COALESCE(${submissions.totalTokens}, 0)`,
      totalCost: sql<number>`COALESCE(CAST(${submissions.totalCost} AS DECIMAL(12,4)), 0)`,
      submissionCount: sql<number>`COALESCE(${submissions.submitCount}, 0)`,
      updatedAt: submissions.updatedAt,
    })
    .from(users)
    .leftJoin(submissions, eq(submissions.userId, users.id))
    .where(eq(users.username, username))
    .limit(1);

  if (!result) {
    return null;
  }

  let rank: number | null = null;

  const rankingValue = sortBy === "cost" ? Number(result.totalCost) || 0 : Number(result.totalTokens) || 0;

  if (rankingValue > 0) {
    const rankResult = await db.execute<{ rank: number }>(sql`
      WITH ranked AS (
        SELECT
          user_id,
          RANK() OVER (
            ORDER BY
              ${sortBy === "cost"
                ? sql`CAST(total_cost AS DECIMAL(12,4)) DESC, total_tokens DESC`
                : sql`total_tokens DESC, CAST(total_cost AS DECIMAL(12,4)) DESC`}
          ) AS rank
        FROM submissions
      )
      SELECT rank FROM ranked WHERE user_id = ${result.id}
    `);

    rank = (rankResult as unknown as { rank: number }[])[0]?.rank || null;
  }

  return {
    user: {
      id: result.id,
      username: result.username,
      displayName: result.displayName,
      avatarUrl: result.avatarUrl,
    },
    stats: {
      totalTokens: Number(result.totalTokens) || 0,
      totalCost: Number(result.totalCost) || 0,
      submissionCount: Number(result.submissionCount) || 0,
      rank,
      updatedAt: result.updatedAt?.toISOString() || null,
    },
  };
}

export function getUserEmbedStats(username: string, sortBy: EmbedSortBy = "tokens"): Promise<UserEmbedStats | null> {
  return unstable_cache(
    () => fetchUserEmbedStats(username, sortBy),
    [`embed-user:${username}:${sortBy}`],
    {
      tags: [`user:${username}`, `embed-user:${username}`, `embed-user:${username}:${sortBy}`],
      revalidate: 60,
    }
  )();
}
