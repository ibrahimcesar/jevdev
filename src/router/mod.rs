//! Cost-priced, security-aware model routing.
//!
//! Routing priced per token is what fails; routing priced per context rebuild
//! is what works. Both formulas from the design notes are here, with the two
//! changes that make the routed path viable: the worker receives a purpose-built
//! context (`x_small`) and the frontier model rereads a scored summary (`back`)
//! rather than everything the worker produced.

use crate::config::{Config, ModelSpec, SecurityConfig, Tier, Trust};
use crate::jev::{questions, Jev};
use crate::state::Sensitivity;
use anyhow::{Context as _, Result};
use globset::{Glob, GlobSet, GlobSetBuilder};
use serde_json::json;
use std::path::Path;

/// USD per million tokens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Price {
    pub input: f64,
    pub cached: f64,
    pub output: f64,
}

const M: f64 = 1_000_000.0;

/// Pure frontier: generate `y`, read `z`. The paper's `25Y + 5Z`.
pub fn frontier_cost(y: f64, z: f64, f: &Price) -> f64 {
    (f.output * y + f.input * z) / M
}

/// Routed: the worker loads `x_small`, generates `y`, reads `z`; the frontier
/// model then reads `back` tokens of scored result. The paper's
/// `3X + 20Y + 8Z` becomes small when `x_small` and `back` are small.
pub fn routed_cost(x_small: f64, y: f64, z: f64, back: f64, w: &Price, f: &Price) -> f64 {
    (w.input * (x_small + z) + w.output * y + f.input * back) / M
}

#[derive(Clone, Debug)]
pub struct CostRow {
    pub model: String,
    pub tier: Tier,
    pub est: f64,
    pub note: String,
}

#[derive(Clone, Debug)]
pub struct RoutePlan {
    pub model: String,
    pub tier: Tier,
    pub est_cost: f64,
    pub p: f64,
    pub sensitivity: Sensitivity,
    pub alternatives: Vec<CostRow>,
    pub reason: String,
}

/// Maps paths to data tiers (Table IV) and tiers to eligible providers.
pub struct SensitivityPolicy {
    open: GlobSet,
    restricted: GlobSet,
    custom: GlobSet,
    custom_excludes: Vec<String>,
}

fn globs(patterns: &[String]) -> Result<GlobSet> {
    let mut b = GlobSetBuilder::new();
    for p in patterns {
        b.add(Glob::new(p).with_context(|| format!("bad glob {p}"))?);
    }
    Ok(b.build()?)
}

impl SensitivityPolicy {
    pub fn new(cfg: &SecurityConfig) -> Result<Self> {
        Ok(Self {
            open: globs(&cfg.open)?,
            restricted: globs(&cfg.restricted)?,
            custom: globs(&cfg.custom)?,
            custom_excludes: cfg.custom_excludes.clone(),
        })
    }

    pub fn classify_path(&self, p: &Path) -> Sensitivity {
        if self.custom.is_match(p) {
            Sensitivity::Custom
        } else if self.restricted.is_match(p) {
            Sensitivity::Restricted
        } else if self.open.is_match(p) {
            Sensitivity::Open
        } else {
            Sensitivity::Standard
        }
    }

    /// The most restrictive tier over `paths`, or `None` when there are none.
    pub fn classify<'a>(&self, paths: impl IntoIterator<Item = &'a Path>) -> Option<Sensitivity> {
        paths.into_iter().map(|p| self.classify_path(p)).max()
    }

    pub fn eligible(&self, m: &ModelSpec, s: Sensitivity) -> bool {
        match s {
            Sensitivity::Open => true,
            Sensitivity::Standard => matches!(m.trust, Trust::FirstParty | Trust::Vetted),
            Sensitivity::Restricted => matches!(m.trust, Trust::FirstParty),
            Sensitivity::Custom => !self.custom_excludes.iter().any(|v| m.provider.eq_ignore_ascii_case(v) || m.id.contains(v.as_str())),
        }
    }
}

pub struct Router {
    models: Vec<ModelSpec>,
    policy: SensitivityPolicy,
    frontier: String,
    routing: bool,
}

impl Router {
    pub fn new(cfg: &Config) -> Result<Self> {
        Ok(Self {
            models: cfg.models.clone(),
            policy: SensitivityPolicy::new(&cfg.security)?,
            frontier: cfg.llm.frontier.clone(),
            routing: cfg.llm.routing,
        })
    }

    pub fn policy(&self) -> &SensitivityPolicy {
        &self.policy
    }

    pub fn frontier(&self) -> &ModelSpec {
        self.models.iter().find(|m| m.id == self.frontier).unwrap_or(&self.models[0])
    }

    pub fn model(&self, id: &str) -> Option<&ModelSpec> {
        self.models.iter().find(|m| m.id == id)
    }

    pub fn eligible(&self, s: Sensitivity) -> Vec<&ModelSpec> {
        self.models.iter().filter(|m| self.policy.eligible(m, s)).collect()
    }

    /// Estimated cost per model for a call with `ctx` context tokens, expecting
    /// `out` generated tokens and `reads` tokens of tool output.
    pub fn costs(&self, ctx: u32, out: u32, reads: u32, s: Sensitivity) -> Vec<CostRow> {
        let f = self.frontier().price();
        let back = (out as f64 / 4.0).max(200.0);
        self.eligible(s)
            .into_iter()
            .map(|m| {
                let est = if m.tier == Tier::Frontier {
                    (f.input * ctx as f64) / M + frontier_cost(out as f64, reads as f64, &m.price())
                } else {
                    routed_cost(ctx as f64, out as f64, reads as f64, back, &m.price(), &f)
                };
                let note = if m.tier == Tier::Frontier { "stays on frontier".to_string() } else { format!("purpose-built ctx {ctx} tok, {back:.0} tok back") };
                CostRow { model: m.id.clone(), tier: m.tier, est, note }
            })
            .collect()
    }

    /// Choose a model for a subtask. Sensitivity filters, the cost model prices,
    /// Jev chooses, and a low-probability choice falls back to the frontier.
    pub async fn choose(&self, subtask: &str, ctx: u32, out: u32, reads: u32, s: Sensitivity, jev: &Jev) -> Result<RoutePlan> {
        let alternatives = self.costs(ctx, out, reads, s);
        let frontier = self.frontier();
        let frontier_cost = alternatives.iter().find(|r| r.model == frontier.id).map(|r| r.est).unwrap_or(0.0);
        let mut plan = RoutePlan {
            model: frontier.id.clone(),
            tier: Tier::Frontier,
            est_cost: frontier_cost,
            p: 1.0,
            sensitivity: s,
            alternatives: alternatives.clone(),
            reason: "routing disabled".into(),
        };
        if !self.routing {
            return Ok(plan);
        }
        let eligible: Vec<&ModelSpec> = self.eligible(s).into_iter().filter(|m| m.tier != Tier::Cheap).collect();
        if eligible.len() <= 1 {
            plan.reason = format!("only {} is eligible for {s} data", frontier.id);
            return Ok(plan);
        }
        let costs: serde_json::Map<String, serde_json::Value> = alternatives.iter().map(|r| (r.model.clone(), json!(r.est))).collect();
        let (id, q) = questions::route(&eligible);
        let state = json!({
            "subtask": subtask,
            "context_tokens": ctx,
            "expected_output_tokens": out,
            "expected_read_tokens": reads,
            "sensitivity": s.to_string(),
            "costs": costs,
        });
        let answer = jev.ask_one(state, &id, q).await?;
        let (choice, p) = answer.as_choice().unwrap_or((frontier.id.as_str(), 0.0));
        let chosen = eligible.iter().find(|m| m.id == choice).copied();
        match chosen {
            Some(m) if p >= 0.5 => {
                plan.model = m.id.clone();
                plan.tier = m.tier;
                plan.est_cost = alternatives.iter().find(|r| r.model == m.id).map(|r| r.est).unwrap_or(0.0);
                plan.p = p;
                plan.reason = format!("jev chose {} (p={p:.2}); est ${:.4} vs frontier ${:.4}", m.id, plan.est_cost, frontier_cost);
            }
            Some(m) => {
                plan.p = p;
                plan.reason = format!("jev leaned {} at p={p:.2}, below 0.5; staying on frontier", m.id);
            }
            None => {
                plan.reason = "jev choice not eligible; staying on frontier".into();
            }
        }
        Ok(plan)
    }
}

/// The paper's arithmetic, for `jevdev cost`. `x`, `y`, `z` are in millions of tokens.
pub fn cost_table(x: f64, y: f64, z: f64, f: &Price, w: &Price) -> String {
    let pure = f.output * y + f.input * z;
    let load = w.input * x;
    let gen = w.output * y;
    let reads = w.input * z;
    let reload = f.input * (y + z);
    let routed = load + gen + reads + reload;
    let x_small = x * 0.1;
    let back = y * 0.25;
    let smart = w.input * (x_small + z) + w.output * y + f.input * back;
    let mut s = String::new();
    s.push_str(&format!("session shape: X={x:.2}M context, Y={y:.2}M generated, Z={z:.2}M read\n\n"));
    s.push_str(&format!("Path 1  pure frontier\n  generate {:>6.2}   read {:>6.2}\n  total   {pure:>6.2}\n\n", f.output * y, f.input * z));
    s.push_str(&format!(
        "Path 2  frontier → worker → frontier (full transcript)\n  worker loads context {load:>6.2}\n  worker generates     {gen:>6.2}\n  worker reads         {reads:>6.2}\n  frontier reloads     {reload:>6.2}\n  total                {routed:>6.2}\n\n"
    ));
    s.push_str(&format!(
        "Path 3  frontier → worker → frontier (jev harness)\n  worker gets a purpose-built context of {x_small:.3}M and returns a scored chunk of {back:.3}M\n  total                {smart:>6.2}\n\n"
    ));
    s.push_str(&format!(
        "routing the naive way costs {:.0}% of pure frontier; the harness way costs {:.0}%\n",
        routed / pure * 100.0,
        smart / pure * 100.0
    ));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paper_numbers_reproduce() {
        let opus = Price { input: 5.0, cached: 0.5, output: 25.0 };
        let sonnet = Price { input: 3.0, cached: 0.3, output: 15.0 };
        let (x, y, z) = (0.65, 0.12, 0.23);
        let pure = opus.output * y + opus.input * z;
        let routed = sonnet.input * x + sonnet.output * y + sonnet.input * z + opus.input * (y + z);
        assert!((pure - 4.15).abs() < 0.01);
        assert!((routed - 6.19).abs() < 0.01);
    }

    #[test]
    fn sensitivity_orders_and_filters() {
        let cfg = SecurityConfig::default();
        let p = SensitivityPolicy::new(&cfg).unwrap();
        assert_eq!(p.classify_path(Path::new("src/main.rs")), Sensitivity::Standard);
        assert_eq!(p.classify_path(Path::new("docs/guide.md")), Sensitivity::Open);
        assert_eq!(p.classify_path(Path::new("config/.env.local")), Sensitivity::Restricted);
        let paths = [Path::new("docs/a.md"), Path::new(".env")];
        assert_eq!(p.classify(paths.iter().copied()), Some(Sensitivity::Restricted));
    }
}
