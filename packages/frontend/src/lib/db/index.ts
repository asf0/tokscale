import { drizzle } from "drizzle-orm/postgres-js";
import * as schema from "./schema";

const connectionString = process.env.DATABASE_URL;

if (!connectionString) {
  throw new Error("DATABASE_URL environment variable is not set");
}

// Singleton pattern: prevent creating multiple connection pools across
// serverless invocations sharing the same runtime (hot-start reuse).
//
// Use drizzle's config-based API to create the postgres client internally.
// Passing a `postgres` Sql instance directly causes type errors on Vercel
// due to duplicate package resolution in the monorepo (two copies of postgres
// with incompatible branded types).
function createDb() {
  return drizzle({
    connection: {
      url: connectionString!,
      ssl: process.env.NODE_ENV === "production" ? "require" : false,

      // Serverless-optimized pool settings:
      // Each Vercel function instance gets its own pool. With dozens of
      // concurrent cold-starts, max:5 per instance quickly exceeds the
      // database server's max_connections (error 53300).
      max: 1,

      // Close idle connections after 20 s so they don't linger between
      // infrequent invocations.
      idle_timeout: 20,

      // Hard cap: recycle every connection after 5 minutes regardless of
      // activity. Prevents stale connections after deploys / DB restarts.
      max_lifetime: 60 * 5,

      // Fail fast when the DB is unreachable instead of hanging the request.
      connect_timeout: 10,

      // Prepared statements are connection-scoped. In serverless, the
      // connection that prepared a statement may be gone by the next
      // invocation, causing "prepared statement does not exist" errors.
      prepare: false,
    },
    schema,
  });
}

const globalForDb = globalThis as unknown as {
  _db: ReturnType<typeof createDb> | undefined;
};

if (!globalForDb._db) {
  globalForDb._db = createDb();
}

export const db = globalForDb._db;

export * from "./schema";
