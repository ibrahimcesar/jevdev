//! Anthropic Messages API over raw HTTP (Rust has no official SDK).
//!
//! Stable content goes first and carries `cache_control`, so a turn whose
//! assembled context keeps the previous prefix reads it from cache.

use super::{Completion, LlmClient, LlmUsage, Prompt};
use anyhow::{anyhow, Context as _, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::time::Duration;

const API_VERSION: &str = "2023-06-01";
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
const OAUTH_BETA: &str = "oauth-2025-04-20";

enum Auth {
    ApiKey(String),
    Bearer(String),
}

pub struct AnthropicClient {
    http: reqwest::Client,
    base: String,
    auth: Auth,
    fallbacks: bool,
}

impl AnthropicClient {
    /// Credentials, in order: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, then a
    /// short-lived token from `ant auth print-credentials --access-token`.
    pub fn from_env(fallbacks: bool) -> Result<Self> {
        let base = std::env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| "https://api.anthropic.com".into());
        let api_key = std::env::var("ANTHROPIC_API_KEY").ok().map(|k| k.trim().to_string()).filter(|k| !k.is_empty());
        let auth = if let Some(k) = api_key {
            Auth::ApiKey(k)
        } else if let Some(t) = std::env::var("ANTHROPIC_AUTH_TOKEN").ok().map(|t| t.trim().to_string()).filter(|t| !t.is_empty()) {
            Auth::Bearer(t)
        } else {
            let out = std::process::Command::new("ant").args(["auth", "print-credentials", "--access-token"]).output();
            match out {
                Ok(o) if o.status.success() => {
                    let t = String::from_utf8_lossy(&o.stdout).trim().to_string();
                    if t.is_empty() {
                        return Err(anyhow!("no Anthropic credentials: set ANTHROPIC_API_KEY or run `ant auth login`"));
                    }
                    Auth::Bearer(t)
                }
                _ => return Err(anyhow!("no Anthropic credentials: set ANTHROPIC_API_KEY or run `ant auth login`")),
            }
        };
        let http = reqwest::Client::builder().timeout(Duration::from_secs(600)).build()?;
        Ok(Self { http, base: base.trim_end_matches('/').to_string(), auth, fallbacks })
    }

    fn supports_effort(model: &str) -> bool {
        !model.starts_with("claude-haiku") && !model.contains("4-5")
    }

    fn supports_fallbacks(model: &str) -> bool {
        model.starts_with("claude-opus-5") || model.starts_with("claude-fable")
    }

    async fn messages(&self, body: Value, betas: &[&str]) -> Result<Value> {
        let mut delay = Duration::from_secs(2);
        let mut last = None;
        for attempt in 0..4 {
            let mut req = self.http.post(format!("{}/v1/messages", self.base)).header("anthropic-version", API_VERSION).header("content-type", "application/json");
            let mut beta_list: Vec<&str> = betas.to_vec();
            req = match &self.auth {
                Auth::ApiKey(k) => req.header("x-api-key", k),
                Auth::Bearer(t) => {
                    beta_list.push(OAUTH_BETA);
                    req.bearer_auth(t)
                }
            };
            if !beta_list.is_empty() {
                req = req.header("anthropic-beta", beta_list.join(","));
            }
            match req.json(&body).send().await {
                Ok(r) => {
                    let status = r.status();
                    let text = r.text().await.unwrap_or_default();
                    if status.is_success() {
                        return serde_json::from_str(&text).context("decoding messages response");
                    }
                    let retry = status.as_u16() == 429 || status.as_u16() == 529 || status.is_server_error();
                    last = Some(anyhow!("anthropic {}: {}", status, crate::state::truncate(&text, 600)));
                    if !retry {
                        break;
                    }
                }
                Err(e) => last = Some(anyhow!("anthropic request failed: {e}")),
            }
            if attempt < 3 {
                tokio::time::sleep(delay).await;
                delay *= 2;
            }
        }
        Err(last.unwrap_or_else(|| anyhow!("anthropic request failed")))
    }

    fn parse(model: &str, v: &Value) -> Result<Completion> {
        let stop = v.get("stop_reason").and_then(Value::as_str).unwrap_or("").to_string();
        if stop == "refusal" {
            let details = v.get("stop_details").map(|d| d.to_string()).unwrap_or_default();
            return Err(anyhow!("model refused: {details}"));
        }
        let text: String = v
            .get("content")
            .and_then(Value::as_array)
            .map(|blocks| blocks.iter().filter(|b| b.get("type").and_then(Value::as_str) == Some("text")).filter_map(|b| b.get("text").and_then(Value::as_str)).collect::<Vec<_>>().join("\n"))
            .unwrap_or_default();
        let u = v.get("usage").cloned().unwrap_or(Value::Null);
        let g = |k: &str| u.get(k).and_then(Value::as_u64).unwrap_or(0);
        let usage = LlmUsage { input: g("input_tokens"), cached_read: g("cache_read_input_tokens"), cache_write: g("cache_creation_input_tokens"), output: g("output_tokens") };
        Ok(Completion { text, usage, stop, model: v.get("model").and_then(Value::as_str).unwrap_or(model).to_string() })
    }
}

#[async_trait]
impl LlmClient for AnthropicClient {
    async fn complete(&self, model: &str, prompt: &Prompt, max_tokens: u32, effort: &str) -> Result<Completion> {
        let mut system = vec![json!({ "type": "text", "text": prompt.system, "cache_control": { "type": "ephemeral" } })];
        if !prompt.pinned.trim().is_empty() {
            system.push(json!({ "type": "text", "text": prompt.pinned }));
        }
        let mut content = Vec::new();
        if !prompt.context.trim().is_empty() {
            content.push(json!({ "type": "text", "text": prompt.context, "cache_control": { "type": "ephemeral" } }));
        }
        content.push(json!({ "type": "text", "text": prompt.query }));
        let mut body = json!({
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{ "role": "user", "content": content }],
        });
        if Self::supports_effort(model) {
            body["output_config"] = json!({ "effort": effort });
        }
        let mut betas = Vec::new();
        if self.fallbacks && Self::supports_fallbacks(model) {
            body["fallbacks"] = json!("default");
            betas.push(FALLBACK_BETA);
        }
        let v = self.messages(body, &betas).await?;
        Self::parse(model, &v)
    }

    async fn small(&self, model: &str, system: &str, user: &str, max_tokens: u32) -> Result<String> {
        let mut body = json!({
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{ "role": "user", "content": user }],
        });
        if Self::supports_effort(model) {
            body["output_config"] = json!({ "effort": "low" });
        }
        let v = self.messages(body, &[]).await?;
        Ok(Self::parse(model, &v)?.text)
    }

    fn name(&self) -> &'static str {
        "anthropic"
    }
}
