# Setup guide — deploying this for real

This walks through every step that needs a human and an account. The code is done; this is the
"get it live" part. Take it slow, do one part at a time. Anywhere you see `ALL_CAPS` in a command,
replace it with your value.

Rough order: GitHub repo → GCP project + billing → keyless auth → secrets → Grafana → Langfuse →
deploy → verify → benchmarks.

You'll need (all free to start): a GitHub account, a Google account, a Grafana Cloud account, a
Langfuse account. The only thing that can cost money is the GPU on Cloud Run, covered by the GCP
$300 credit (see Part 2).

---

## Part 0 — Try it locally first (optional, 5 min)

Proves the app works before you touch any cloud.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn tools.fake_vllm:app --port 8000 &              # fake model
DEMO_API_KEY=demokey uvicorn app.main:app --port 8080  # the gateway
```

Open http://localhost:8080 and chat (no key needed). Ctrl+C when done.

---

## Part 1 — GitHub repo

1. Create a new repo on GitHub named `llm-inference-service` (empty, no README).
2. In this project, find-and-replace the placeholders:
   - `ShibaWang-1028` → your GitHub username (in `README.md`).
   - `YOUR_SERVICE_URL` → leave for now, you'll fill it after the first deploy.
3. Push:
   ```bash
   git add -A
   git commit -m "Initial commit: LLM inference service"
   git branch -M main
   git remote add origin https://github.com/ShibaWang-1028/llm-inference-service.git
   git push -u origin main
   ```
   The CI will run lint + tests on this push (the deploy jobs are skipped until GCP is wired up, and
   will error on the missing variables until Part 8, which is fine for now).

---

## Part 2 — GCP project + billing (this is the one with the money caveat)

You can run all `gcloud` commands from [Cloud Shell](https://shell.cloud.google.com) (no install) or
install the [gcloud CLI](https://cloud.google.com/sdk/docs/install) locally.

1. Create a project and note its ID:
   ```bash
   gcloud projects create YOUR_PROJECT_ID --name="LLM Inference"
   gcloud config set project YOUR_PROJECT_ID
   ```
2. **Upgrade to a paid account.** Cloud Run GPUs are blocked on the free trial. In the Console go to
   Billing, link a billing account, and **upgrade/activate the full account** (add a payment method).
   Your remaining $300 credit still applies to paid usage, so this is still ~$0 out of pocket for a
   while, but real charges are possible after the credit, which is why you set budget alerts below.
3. Enable the APIs:
   ```bash
   gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
     cloudbuild.googleapis.com secretmanager.googleapis.com \
     iamcredentials.googleapis.com sts.googleapis.com
   ```
4. **Budget alerts** (so you're never surprised): Console → Billing → Budgets & alerts → Create
   budget. Set a small monthly amount (e.g. $20) with alerts at 50/80/100%.
5. **Region + GPU quota.** Use `us-central1` (L4 available). New paid projects get a default quota of
   3 non-redundant L4s per region, which is plenty. If you ever need more: Console → IAM & Admin →
   Quotas → search "NvidiaL4GpuAllocNoZonalRedundancyPerProjectRegion" → request an increase.

---

## Part 3 — Keyless auth for GitHub Actions (Workload Identity Federation)

This lets the CI deploy to GCP without storing a long-lived key. Run these once (replace the repo).

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
export GH_REPO="ShibaWang-1028/llm-inference-service"
export GH_OWNER="ShibaWang-1028"

# 1) Identity pool + GitHub OIDC provider (the attribute-condition locks it to your account)
gcloud iam workload-identity-pools create github-pool --location=global \
  --display-name="GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global --workload-identity-pool=github-pool \
  --display-name="GitHub" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner == '${GH_OWNER}'"

# 2) Let your repo's identity act, and grant the roles the pipeline needs
export PRINCIPAL="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${GH_REPO}"
for ROLE in roles/run.admin roles/cloudbuild.builds.editor roles/artifactregistry.admin \
            roles/storage.admin roles/iam.serviceAccountUser \
            roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding $PROJECT_ID --member="$PRINCIPAL" --role="$ROLE"
done

# 3) The value you put into GitHub as WIF_PROVIDER:
echo "projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
```

Save that last printed string.

---

## Part 4 — Secrets in Secret Manager

The Cloud Run runtime reads these. Create them now.

```bash
# 1) Gateway API key — generate a random one and keep a copy
GATEWAY_KEY=$(openssl rand -hex 24)
echo "Your gateway API key (save it): $GATEWAY_KEY"
printf '%s' "$GATEWAY_KEY" | gcloud secrets create gateway-api-key --data-file=-

# 2) OTel collector config — fill it in after Part 5, then run:
#    gcloud secrets create otel-collector-config --data-file=otel-collector.local.yaml

# Let Cloud Run's runtime service account read the secrets:
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

(Optional Langfuse secrets are created in Part 6.)

---

## Part 5 — Grafana Cloud

1. Sign up at [grafana.com](https://grafana.com) (free tier).
2. In your stack: Connections → Prometheus (or "Sending metrics") → copy the **remote-write URL**,
   your **instance/user ID** (numeric), and create an **access policy token** with the
   `metrics:write` (MetricsPublisher) scope.
3. Make your collector config:
   ```bash
   cp monitoring/otel-collector.yaml otel-collector.local.yaml
   ```
   Edit `otel-collector.local.yaml` and fill the three placeholders (URL, instance ID, token). This
   file is gitignored, do not commit it.
4. Upload it as the secret:
   ```bash
   gcloud secrets create otel-collector-config --data-file=otel-collector.local.yaml
   ```
5. Import the dashboard: Grafana → Dashboards → New → Import → upload `monitoring/grafana-dashboard.json`,
   choose your Prometheus data source.
6. Create the two alerts from `monitoring/alerts.yaml` (or recreate them in the UI using the PromQL in
   [monitoring.md](monitoring.md)).

---

## Part 6 — Langfuse (optional but recommended for the cost story)

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com), create a project, copy the public +
   secret keys.
2. Store them and grant access (the secretAccessor binding from Part 4 already covers them):
   ```bash
   printf '%s' "pk-lf-..." | gcloud secrets create langfuse-public --data-file=-
   printf '%s' "sk-lf-..." | gcloud secrets create langfuse-secret --data-file=-
   ```
3. In `deploy/cloudrun.yaml`, set `ENABLE_LANGFUSE` to `"true"` and uncomment the two
   `LANGFUSE_*` secret blocks. Commit and push.
4. Register the model price: Langfuse → Settings → Models → New model, `match_pattern`
   `(?i)^qwen2\.5-7b-instruct$`, with per-token input/output prices (see [monitoring.md](monitoring.md)).

---

## Part 7 — GitHub repo variables and secrets

In your repo → Settings → Secrets and variables → Actions:

- **Variables** tab:
  - `GCP_PROJECT_ID` = your project ID
  - `WIF_PROVIDER` = the string printed at the end of Part 3
- **Secrets** tab:
  - `GATEWAY_API_KEY` = the same key you generated in Part 4

---

## Part 8 — Deploy

Push to `main` (or re-run the latest Action). The pipeline will: build the image with Cloud Build,
deploy to Cloud Run with the L4 GPU, smoke-test the live endpoint, and run the k6 gate.

```bash
git commit --allow-empty -m "Trigger deploy" && git push
```

Watch it under the repo's Actions tab. The first build is slow (large CUDA image). Once green, get
your URL:

```bash
gcloud run services describe llm-inference --region=us-central1 --format='value(status.url)'
```

Put that URL into `README.md` (`YOUR_SERVICE_URL`) and commit.

---

## Part 9 — Verify

```bash
URL=https://YOUR_SERVICE_URL.run.app
# readiness (first call cold-starts the GPU, may take ~30-60s)
curl "$URL/health/ready"
# a streamed completion
curl -N "$URL/v1/chat/completions" \
  -H "Authorization: Bearer YOUR_GATEWAY_KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"hello"}],"stream":true}'
```

Then: open the URL in a browser (chat UI), generate some traffic, and confirm the Grafana panels
move and a Langfuse trace shows tokens + cost.

---

## Part 10 — Run the benchmarks (for the before→after table)

The benchmarks need a GPU. Cheapest reliable way is a short-lived L4 VM:

```bash
gcloud compute instances create bench-l4 --zone=us-central1-a \
  --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE --provisioning-model=SPOT \
  --image-family=common-cu123 --image-project=deeplearning-platform-release --boot-disk-size=100GB
# SSH in, clone the repo, then follow docs/benchmarks.md (install requirements-bench.txt and run the 3 configs)
# DELETE IT WHEN DONE so it stops billing:
gcloud compute instances delete bench-l4 --zone=us-central1-a
```

Run `python -m benchmarks.plot`, paste `benchmarks/results/summary.md` into
[docs/benchmarks.md](benchmarks.md) and the README table, and commit `comparison.png`.

---

## Cost control + teardown

- Idle cost is ~$0 (scale-to-zero). You only pay while an instance is actively serving.
- Keep `max-instances` low (it's 3 in `cloudrun.yaml`) and the budget alerts on.
- To stop everything: the service already scales to zero on its own. To remove it entirely:
  ```bash
  gcloud run services delete llm-inference --region=us-central1
  ```
- Delete any benchmark VM as soon as you're done with it.

If any `gcloud` step fails with a permission error, the message names the missing role; add it to the
`PRINCIPAL` (Part 3) or the runtime SA (Part 4) and retry.
