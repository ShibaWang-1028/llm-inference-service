// k6 load test used as the CI deploy gate.
//
// If p95 latency or the error rate breach the thresholds, k6 exits non-zero (99)
// and the GitHub Actions step fails, so a bad build never stays deployed.
//
// Run locally:
//   URL=https://your-service.run.app API_KEY=yourkey k6 run benchmarks/loadtest.js
//
// Tunables via env: P95_MS, ERROR_RATE, VUS, MODEL.

import http from "k6/http";
import { check } from "k6";

const URL = __ENV.URL;
const API_KEY = __ENV.API_KEY || "";
const MODEL = __ENV.MODEL || "Qwen2.5-7B-Instruct";
const P95_MS = Number(__ENV.P95_MS || 8000);
const ERROR_RATE = Number(__ENV.ERROR_RATE || 0.02);
const VUS = Number(__ENV.VUS || 5);

export const options = {
  scenarios: {
    load: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "20s", target: VUS },
        { duration: "40s", target: VUS },
        { duration: "10s", target: 0 },
      ],
      gracefulStop: "30s",
    },
  },
  thresholds: {
    // <2% failed requests, p95 under the budget (defaults; tune after measuring)
    http_req_failed: [`rate<${ERROR_RATE}`],
    http_req_duration: [`p(95)<${P95_MS}`],
  },
};

const headers = { "Content-Type": "application/json" };
if (API_KEY) headers["Authorization"] = `Bearer ${API_KEY}`;

export default function () {
  const payload = JSON.stringify({
    model: MODEL,
    messages: [{ role: "user", content: "In one sentence, what is continuous batching?" }],
    max_tokens: 32,
    temperature: 0,
  });

  const res = http.post(`${URL}/v1/chat/completions`, payload, { headers, timeout: "60s" });

  check(res, {
    "status is 200": (r) => r.status === 200,
    "has choices": (r) => {
      try {
        return JSON.parse(r.body).choices.length > 0;
      } catch (e) {
        return false;
      }
    },
  });
}
