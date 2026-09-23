//! The Jev client.
//!
//! Jev is TypeSafe's System One model: it takes a `state` plus a map of typed
//! questions and returns typed answers with probabilities, in one request, with
//! every question evaluated in parallel. The harness never parses prose from it.
//!
//! Wire format (`POST /v1/systemone`, `Authorization: Bearer $TYPESAFE_API_KEY`):
//!
//! ```json
//! { "state": {...}, "model": "jev-latest",
//!   "questions": { "id": { "type": "choice", "instructions": "...", "criteria": {"a": "..."} } } }
//! ```
//!
//! [`questions`] holds the harness's own question builders; [`local`] is an
//! offline heuristic transport so the harness runs without a key.

pub mod http;
pub mod local;
pub mod questions;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// One typed question. `instructions` may be a string, object, or array.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// Pick one option. Up to 255 options; descriptions may be null.
    Choice { instructions: Value, criteria: BTreeMap<String, Option<String>> },
    /// Rate on an ordered scale of 2 to 10 level descriptions, low to high.
    Score { instructions: Value, criteria: Vec<String> },
    /// Yes or no, returned as the probability of yes.
    Noul {
        instructions: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NoulCriteria {
    #[serde(rename = "true")]
    pub yes: String,
    #[serde(rename = "false")]
    pub no: String,
}

impl Question {
    pub fn choice<I, K, D>(instructions: impl Into<Value>, options: I) -> Self
    where
        I: IntoIterator<Item = (K, D)>,
        K: Into<String>,
        D: Into<String>,
    {
        Self::Choice {
            instructions: instructions.into(),
            criteria: options.into_iter().map(|(k, d)| (k.into(), Some(d.into()))).collect(),
        }
    }
    pub fn choice_bare<I, K>(instructions: impl Into<Value>, options: I) -> Self
    where
        I: IntoIterator<Item = K>,
        K: Into<String>,
    {
        Self::Choice { instructions: instructions.into(), criteria: options.into_iter().map(|k| (k.into(), None)).collect() }
    }
    pub fn score<I, L>(instructions: impl Into<Value>, levels: I) -> Self
    where
        I: IntoIterator<Item = L>,
        L: Into<String>,
    {
        Self::Score { instructions: instructions.into(), criteria: levels.into_iter().map(Into::into).collect() }
    }
    pub fn noul(instructions: impl Into<Value>) -> Self {
        Self::Noul { instructions: instructions.into(), criteria: None }
    }
    pub fn noul_with(instructions: impl Into<Value>, yes: &str, no: &str) -> Self {
        Self::Noul { instructions: instructions.into(), criteria: Some(NoulCriteria { yes: yes.into(), no: no.into() }) }
    }
    pub fn kind(&self) -> &'static str {
        match self {
            Question::Choice { .. } => "choice",
            Question::Score { .. } => "score",
            Question::Noul { .. } => "noul",
        }
    }
    pub fn instructions_text(&self) -> String {
        let v = match self {
            Question::Choice { instructions, .. } | Question::Score { instructions, .. } | Question::Noul { instructions, .. } => instructions,
        };
        match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemOneRequest {
    pub state: Value,
    pub model: String,
    pub questions: BTreeMap<String, Question>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Choice {
        choice: String,
        #[serde(default)]
        confidence: f64,
        #[serde(default)]
        probabilities: BTreeMap<String, f64>,
    },
    Score {
        score: f64,
        #[serde(default)]
        confidence: f64,
        #[serde(default)]
        probabilities: BTreeMap<String, f64>,
        #[serde(default)]
        legend: BTreeMap<String, String>,
    },
    Noul {
        noul: f64,
    },
}

impl Answer {
    /// The chosen option and its probability (falls back to confidence).
    pub fn as_choice(&self) -> Option<(&str, f64)> {
        match self {
            Answer::Choice { choice, confidence, probabilities } => {
                let p = probabilities.get(choice).copied().unwrap_or(*confidence);
                Some((choice.as_str(), p))
            }
            _ => None,
        }
    }
    pub fn as_noul(&self) -> Option<f64> {
        match self {
            Answer::Noul { noul } => Some(*noul),
            _ => None,
        }
    }
    /// Score and confidence.
    pub fn as_score(&self) -> Option<(f64, f64)> {
        match self {
            Answer::Score { score, confidence, .. } => Some((*score, *confidence)),
            _ => None,
        }
    }
    /// Options ranked by probability, highest first.
    pub fn ranked(&self) -> Vec<(String, f64)> {
        match self {
            Answer::Choice { probabilities, choice, confidence } => {
                let mut v: Vec<(String, f64)> = probabilities.iter().map(|(k, p)| (k.clone(), *p)).collect();
                if v.is_empty() {
                    v.push((choice.clone(), *confidence));
                }
                v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                v
            }
            Answer::Score { probabilities, .. } => {
                let mut v: Vec<(String, f64)> = probabilities.iter().map(|(k, p)| (k.clone(), *p)).collect();
                v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                v
            }
            Answer::Noul { noul } => vec![("yes".into(), *noul), ("no".into(), 1.0 - noul)],
        }
    }
    pub fn summary(&self) -> String {
        match self {
            Answer::Choice { choice, .. } => {
                let (_, p) = self.as_choice().unwrap();
                format!("{choice} (p={p:.2})")
            }
            Answer::Score { score, confidence, .. } => format!("{score:.2} (conf={confidence:.2})"),
            Answer::Noul { noul } => format!("{}", if *noul >= 0.5 { format!("yes (p={noul:.2})") } else { format!("no (p={:.2})", 1.0 - noul) }),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemOneResponse {
    #[serde(default)]
    pub model: String,
    pub answers: BTreeMap<String, Answer>,
    #[serde(default)]
    pub usage: Usage,
}

#[async_trait]
pub trait Transport: Send + Sync {
    async fn system_one(&self, req: &SystemOneRequest) -> Result<SystemOneResponse>;
    fn name(&self) -> &'static str;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct JevStats {
    pub calls: u64,
    pub questions: u64,
    pub input_tokens: u64,
    pub memo_hits: u64,
}

/// The client. Memoises identical requests, so re-scoring an unchanged chunk
/// for an unchanged query costs nothing.
pub struct Jev {
    transport: Arc<dyn Transport>,
    model: String,
    memo: moka::future::Cache<u64, Arc<SystemOneResponse>>,
    calls: AtomicU64,
    questions: AtomicU64,
    tokens: AtomicU64,
    hits: AtomicU64,
}

impl Jev {
    pub fn new(transport: Arc<dyn Transport>, model: &str) -> Self {
        Self {
            transport,
            model: model.to_string(),
            memo: moka::future::Cache::new(50_000),
            calls: AtomicU64::new(0),
            questions: AtomicU64::new(0),
            tokens: AtomicU64::new(0),
            hits: AtomicU64::new(0),
        }
    }

    /// `auto` picks HTTP when `TYPESAFE_API_KEY` is set, else the local heuristics.
    pub fn from_config(cfg: &crate::config::JevConfig) -> Result<Self> {
        let has_key = std::env::var("TYPESAFE_API_KEY").map(|k| !k.trim().is_empty()).unwrap_or(false);
        let transport: Arc<dyn Transport> = match cfg.transport.as_str() {
            "http" => Arc::new(http::HttpTransport::from_env(&cfg.endpoint)?),
            "local" => Arc::new(local::LocalTransport::default()),
            _ if has_key => Arc::new(http::HttpTransport::from_env(&cfg.endpoint)?),
            _ => Arc::new(local::LocalTransport::default()),
        };
        Ok(Self::new(transport, &cfg.model))
    }

    pub fn transport_name(&self) -> &'static str {
        self.transport.name()
    }

    pub fn stats(&self) -> JevStats {
        JevStats {
            calls: self.calls.load(Ordering::Relaxed),
            questions: self.questions.load(Ordering::Relaxed),
            input_tokens: self.tokens.load(Ordering::Relaxed),
            memo_hits: self.hits.load(Ordering::Relaxed),
        }
    }

    /// One System One call: many questions, one state.
    pub async fn ask(&self, state: Value, questions: BTreeMap<String, Question>) -> Result<Arc<SystemOneResponse>> {
        if questions.is_empty() {
            return Ok(Arc::new(SystemOneResponse { model: self.model.clone(), answers: BTreeMap::new(), usage: Usage::default() }));
        }
        let req = SystemOneRequest { state, model: self.model.clone(), questions };
        let key = {
            let bytes = serde_json::to_vec(&req)?;
            let h = blake3::hash(&bytes);
            u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap())
        };
        if let Some(hit) = self.memo.get(&key).await {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(hit);
        }
        let n = req.questions.len() as u64;
        let resp = Arc::new(self.transport.system_one(&req).await?);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.questions.fetch_add(n, Ordering::Relaxed);
        self.tokens.fetch_add(resp.usage.input_tokens, Ordering::Relaxed);
        self.memo.insert(key, resp.clone()).await;
        Ok(resp)
    }

    /// One question, one answer.
    pub async fn ask_one(&self, state: Value, id: &str, q: Question) -> Result<Answer> {
        let mut qs = BTreeMap::new();
        qs.insert(id.to_string(), q);
        let resp = self.ask(state, qs).await?;
        resp.answers.get(id).cloned().ok_or_else(|| anyhow::anyhow!("jev returned no answer for {id}"))
    }
}

/// Approximate token count of a JSON value, for batching state under Jev's limit.
pub fn value_tokens(v: &Value) -> u32 {
    crate::state::tokens::count(&v.to_string())
}
