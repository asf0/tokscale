import { Navigation } from "@/components/layout/Navigation";
import { LandingPage } from "@/components/landing/LandingPage";
import { getStargazersCount } from "@/lib/github";
import { getLeaderboardData } from "@/lib/leaderboard/getLeaderboard";

export default async function HomePage() {
  const [stargazersCount, topUsersByCost, topUsersByTokens] = await Promise.all([
    getStargazersCount("junhoyeo/tokscale"),
    getLeaderboardData("all", 1, 5, "cost"),
    getLeaderboardData("all", 1, 5, "tokens"),
  ]);

  return (
    <>
      <Navigation />
      <LandingPage
        stargazersCount={stargazersCount}
        topUsersByCost={topUsersByCost.users}
        topUsersByTokens={topUsersByTokens.users}
      />
    </>
  );
}
