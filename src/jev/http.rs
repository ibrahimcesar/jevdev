//! HTTP transport for `https://api.typesafe.ai/v1/systemone`.

use super::{SystemOneRequest, SystemOneResponse, Transport};
use anyhow::{anyhow, Context as _, Result};
use async_trait::async_trait;
use std::time::Duration;

pub struct HttpTransport {
    client: reqwest::Client,
    url: String,
    key: String,
}

impl HttpTransport {
    pub fn from_env(endpoint: &str) -> Result<Self> {
        let key = std::env::var("TYPESAFE_API_KEY").ok().filter(|k| !k.trim().is_empty()).ok_or_else(|| anyhow!("TYPESAFE_API_KEY is not set"))?;
        Ok(Self::new(endpoint, &key))
    }

    pub fn new(endpoint: &str, key: &str) -> Self {
        let client = reqwest::Client::builder().timeout(Duration::from_secs(60)).build().expect("reqwest client");
        Self { client, url: format!("{}/v1/systemone", endpoint.trim_end_matches('/')), key: key.to_string() }
    }
}

#[async_trait]
impl Transport for HttpTransport {
    async fn system_one(&self, req: &SystemOneRequest) -> Result<SystemOneResponse> {
        let mut delay = Duration::from_secs(1);
        let mut last_err = None;
        for attempt in 0..4 {
            let resp = self.client.post(&self.url).bearer_auth(&self.key).json(req).send().await;
            match resp {
                Ok(r) => {
                    let status = r.status();
                    if status.is_success() {
                        return r.json::<SystemOneResponse>().await.context("decoding jev response");
                    }
                    let body = r.text().await.unwrap_or_default();
                    let retryable = status.as_u16() == 429 || status.as_u16() == 529 || status.is_server_error();
                    last_err = Some(anyhow!("jev {}: {}", status, crate::state::truncate(&body, 400)));
                    if !retryable {
                        break;
                    }
                }
                Err(e) => {
                    last_err = Some(anyhow!("jev request failed: {e}"));
                }
            }
            if attempt < 3 {
                tokio::time::sleep(delay).await;
                delay *= 2;
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("jev request failed")))
    }

    fn name(&self) -> &'static str {
        "http"
    }
}
