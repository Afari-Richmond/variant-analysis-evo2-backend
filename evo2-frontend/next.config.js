/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
import "./src/env.js";

/** @type {import("next").NextConfig} */
const config = {
  reactStrictMode: false,

  // This allows the build to succeed even if there are ESLint errors
  eslint: {
    ignoreDuringBuilds: true,
  },

  // This allows the build to succeed even if there are TypeScript type errors
  typescript: {
    ignoreBuildErrors: true,
  },
};

export default config;
