//! Terminal UI.
//!
//! Left: the transcript. Right: Jev's decisions this turn and the context
//! ladder, so you can watch what the model is shown and why. Bottom: usage
//! and the input line, which doubles as the permission prompt.

use crate::config::Config;
use crate::runtime::{emit, ContextRow, Event, EventPermissioner, Session, UsageSnapshot};
use crate::state::{tokens::fmt_k, Visibility};
use anyhow::Result;
use ratatui::crossterm::event::{self as cevent, Event as CEvent, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use ratatui::Frame;
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

struct Pending {
    call: String,
    reasons: Vec<String>,
    reply: oneshot::Sender<bool>,
}

struct App {
    transcript: Vec<Line<'static>>,
    decisions: VecDeque<Line<'static>>,
    rows: Vec<ContextRow>,
    ctx_tokens: u32,
    ctx_budget: u32,
    ctx_note: String,
    usage: UsageSnapshot,
    input: String,
    pending: Option<Pending>,
    busy: bool,
    scroll: u16,
    follow: bool,
    model: String,
    session: String,
    turn: u32,
    banner: String,
    should_quit: bool,
}

impl App {
    fn new(banner: String) -> Self {
        Self {
            transcript: vec![Line::from(Span::styled("jevdev · type a goal and press Enter · Esc quits · y/n answers a permission prompt", Style::default().dim()))],
            decisions: VecDeque::new(),
            rows: Vec::new(),
            ctx_tokens: 0,
            ctx_budget: 0,
            ctx_note: String::new(),
            usage: UsageSnapshot::default(),
            input: String::new(),
            pending: None,
            busy: false,
            scroll: 0,
            follow: true,
            model: String::new(),
            session: String::new(),
            turn: 0,
            banner,
            should_quit: false,
        }
    }

    fn push(&mut self, line: Line<'static>) {
        self.transcript.push(line);
    }

    fn push_text(&mut self, prefix: &'static str, style: Style, text: &str, max_lines: usize) {
        for (i, l) in text.lines().take(max_lines).enumerate() {
            let p = if i == 0 { prefix } else { "      " };
            self.push(Line::from(vec![Span::styled(p, style.add_modifier(Modifier::BOLD)), Span::raw(l.to_string())]));
        }
        if text.lines().count() > max_lines {
            self.push(Line::from(Span::styled(format!("      … {} more lines", text.lines().count() - max_lines), Style::default().dim())));
        }
    }

    fn decision(&mut self, line: Line<'static>) {
        self.decisions.push_back(line);
        while self.decisions.len() > 14 {
            self.decisions.pop_front();
        }
    }

    fn apply(&mut self, e: Event) {
        match e {
            Event::Info(s) => self.push(Line::from(Span::styled(format!("· {s}"), Style::default().dim()))),
            Event::TurnStart { session, turn, query } => {
                self.session = session;
                self.turn = turn;
                self.decisions.clear();
                self.decision(Line::from(Span::styled(format!("turn {turn}"), Style::default().bold())));
                if turn == 1 || self.transcript.len() < 2 {
                    self.push_text("you › ", Style::default().fg(Color::Cyan), &query, 6);
                }
            }
            Event::Decision { point, detail, p } => {
                self.decision(Line::from(vec![Span::styled(format!("{point:<9}"), Style::default().fg(Color::Magenta)), Span::raw(format!("{} ", crate::state::truncate(&detail, 60))), Span::styled(format!("p={p:.2}"), Style::default().dim())]));
            }
            Event::Context { rows, tokens, budget, scored, hidden, dropped, reused } => {
                self.rows = rows;
                self.ctx_tokens = tokens;
                self.ctx_budget = budget;
                self.ctx_note = format!("{scored} scored · {hidden} hidden · {dropped} over budget{}", match reused { Some(true) => " · prefix reused", Some(false) => " · rebuilt", None => "" });
            }
            Event::Route { model, est, p, reason, .. } => {
                self.model = model.clone();
                self.decision(Line::from(vec![Span::styled("route    ", Style::default().fg(Color::Magenta)), Span::styled(model, Style::default().bold()), Span::raw(format!(" ${est:.4} ")), Span::styled(format!("p={p:.2}"), Style::default().dim())]));
                self.decision(Line::from(Span::styled(format!("         {}", crate::state::truncate(&reason, 70)), Style::default().dim())));
            }
            Event::Note { text, .. } => self.push_text("      ", Style::default(), &text, 8),
            Event::ToolPicked { tool, intent, candidates, args } => {
                self.push_text("act › ", Style::default().fg(Color::Yellow), &intent, 4);
                let c: Vec<String> = candidates.iter().map(|(t, p)| format!("{t} {p:.2}")).collect();
                self.decision(Line::from(vec![Span::styled("tool     ", Style::default().fg(Color::Magenta)), Span::styled(tool.clone(), Style::default().bold()), Span::styled(format!("  [{}]", c.join(" · ")), Style::default().dim())]));
                self.push(Line::from(vec![Span::styled("      → ", Style::default().dim()), Span::styled(format!("{tool} {args}"), Style::default().dim())]));
            }
            Event::Permit { verdict, reasons, .. } => {
                let color = match verdict.as_str() {
                    "allow" => Color::Green,
                    "deny" => Color::Red,
                    _ => Color::Yellow,
                };
                self.decision(Line::from(vec![Span::styled("permit   ", Style::default().fg(Color::Magenta)), Span::styled(verdict, Style::default().fg(color).bold()), Span::styled(format!("  {}", crate::state::truncate(&reasons.join("; "), 60)), Style::default().dim())]));
            }
            Event::ToolRan { tool, access, ok, tokens, preview } => {
                let style = if ok { Style::default().fg(Color::Green) } else { Style::default().fg(Color::Red) };
                let head = format!("{tool} · {access} · {} tok{}", fmt_k(tokens), if ok { "" } else { " · failed" });
                self.push(Line::from(vec![Span::styled("tool › ", style.add_modifier(Modifier::BOLD)), Span::styled(head, style)]));
                self.push_text("      ", Style::default().dim(), &preview, 5);
            }
            Event::Subagent { id, goal, status } => self.push(Line::from(vec![Span::styled(format!("{id} › "), Style::default().fg(Color::Blue).bold()), Span::raw(format!("{status}: {}", crate::state::truncate(&goal, 100)))])),
            Event::Done { answer, .. } => {
                self.push_text("done › ", Style::default().fg(Color::Green), &answer, 40);
                self.push(Line::from(""));
                self.busy = false;
            }
            Event::Error(s) => {
                self.push(Line::from(vec![Span::styled("error › ", Style::default().fg(Color::Red).bold()), Span::raw(s)]));
                self.busy = false;
            }
            Event::Usage(u) => self.usage = u,
            Event::Ask { call, reasons, reply } => {
                self.push(Line::from(vec![Span::styled("ask › ", Style::default().fg(Color::Yellow).bold()), Span::raw(format!("{call}  [{}]", reasons.join("; ")))]));
                self.pending = Some(Pending { call, reasons, reply });
            }
        }
        if self.follow {
            self.scroll = u16::MAX;
        }
    }
}

fn vis_bar(v: Visibility) -> (&'static str, Color) {
    match v {
        Visibility::Full => ("████", Color::Green),
        Visibility::Long => ("███ ", Color::Cyan),
        Visibility::Short => ("██  ", Color::Yellow),
        Visibility::Hide => ("·   ", Color::DarkGray),
    }
}

fn draw(f: &mut Frame, app: &mut App) {
    let area = f.area();
    let outer = Layout::default().direction(Direction::Vertical).constraints([Constraint::Length(1), Constraint::Min(5), Constraint::Length(1), Constraint::Length(3)]).split(area);

    let title = Line::from(vec![
        Span::styled(" jevdev ", Style::default().bg(Color::Magenta).fg(Color::Black).bold()),
        Span::raw(format!(" session {} · turn {} · ", if app.session.is_empty() { "–" } else { &app.session }, app.turn)),
        Span::styled(if app.model.is_empty() { "no model yet".to_string() } else { app.model.clone() }, Style::default().bold()),
        Span::styled(format!(" · {}", app.banner), Style::default().dim()),
        if app.busy { Span::styled("  ⏳ working", Style::default().fg(Color::Yellow)) } else { Span::raw("") },
    ]);
    f.render_widget(Paragraph::new(title), outer[0]);

    let main = Layout::default().direction(Direction::Horizontal).constraints([Constraint::Percentage(62), Constraint::Percentage(38)]).split(outer[1]);

    let inner_h = main[0].height.saturating_sub(2) as usize;
    let total = app.transcript.len();
    let max_scroll = total.saturating_sub(inner_h) as u16;
    if app.scroll > max_scroll {
        app.scroll = max_scroll;
    }
    let transcript = Paragraph::new(app.transcript.clone()).wrap(Wrap { trim: false }).scroll((app.scroll, 0)).block(Block::default().borders(Borders::ALL).title(" transcript "));
    f.render_widget(transcript, main[0]);

    let right = Layout::default().direction(Direction::Vertical).constraints([Constraint::Length(16), Constraint::Min(4)]).split(main[1]);
    let decisions: Vec<ListItem> = app.decisions.iter().cloned().map(ListItem::new).collect();
    f.render_widget(List::new(decisions).block(Block::default().borders(Borders::ALL).title(" jev decisions ")), right[0]);

    let ctx_title = format!(" context {} / {} tok · {} ", fmt_k(app.ctx_tokens), fmt_k(app.ctx_budget), app.ctx_note);
    let rows: Vec<ListItem> = app
        .rows
        .iter()
        .map(|r| {
            let (bar, color) = vis_bar(r.visibility);
            ListItem::new(Line::from(vec![
                Span::styled(bar, Style::default().fg(color)),
                Span::styled(format!(" {:<5}", r.visibility), Style::default().fg(color)),
                Span::styled(format!("{:>6} ", fmt_k(r.tokens)), Style::default().dim()),
                Span::styled(format!("{:.2} ", r.p), Style::default().dim()),
                Span::raw(if r.pinned { "📌 " } else { "" }),
                Span::raw(crate::state::truncate(&r.label, 40)),
            ]))
        })
        .collect();
    f.render_widget(List::new(rows).block(Block::default().borders(Borders::ALL).title(ctx_title)), right[1]);

    let u = &app.usage;
    let status = Line::from(vec![
        Span::styled(" llm ", Style::default().bold()),
        Span::raw(format!("in {} · cached {} · out {} · ${:.4}", fmt_k(u.llm.input as u32), fmt_k(u.llm.cached_read as u32), fmt_k(u.llm.output as u32), u.llm_cost)),
        Span::styled("   jev ", Style::default().bold()),
        Span::raw(format!("{} calls · {} questions · {} tok · ${:.4} · {} memo hits", u.jev_calls, u.jev_questions, fmt_k(u.jev_tokens as u32), u.jev_cost, u.memo_hits)),
    ]);
    f.render_widget(Paragraph::new(status).style(Style::default().dim()), outer[2]);

    let (prompt, text, style) = match &app.pending {
        Some(p) => (" permission ", format!("run  {}  ? [y/n]   {}", p.call, crate::state::truncate(&p.reasons.join("; "), 80)), Style::default().fg(Color::Yellow)),
        None if app.busy => (" input ", format!("{}▏  (working… you can queue the next goal)", app.input), Style::default()),
        None => (" input ", format!("{}▏", app.input), Style::default()),
    };
    f.render_widget(Paragraph::new(text).style(style).block(Block::default().borders(Borders::ALL).title(prompt)), outer[3]);
    let _ = Rect::default();
}

/// Run the interactive session.
pub async fn run(cfg: Config, root: &Path, initial_goal: Option<String>) -> Result<()> {
    let (ev_tx, mut ev_rx) = mpsc::unbounded_channel::<Event>();
    let (in_tx, mut in_rx) = mpsc::unbounded_channel::<String>();
    let permissioner = Arc::new(EventPermissioner(ev_tx.clone()));
    let banner = format!("{} · {}", cfg.llm.provider, root.display());
    let root = root.to_path_buf();

    let session_events = ev_tx.clone();
    tokio::spawn(async move {
        let mut session = match Session::open(cfg, &root, session_events.clone(), permissioner).await {
            Ok(s) => s,
            Err(e) => {
                emit(&session_events, Event::Error(format!("could not open session: {e:#}")));
                return;
            }
        };
        while let Some(goal) = in_rx.recv().await {
            if let Err(e) = session.run_goal(&goal).await {
                emit(&session_events, Event::Error(format!("{e:#}")));
            }
        }
    });

    let (key_tx, mut key_rx) = mpsc::unbounded_channel::<CEvent>();
    std::thread::spawn(move || loop {
        if cevent::poll(Duration::from_millis(100)).unwrap_or(false) {
            if let Ok(ev) = cevent::read() {
                if key_tx.send(ev).is_err() {
                    break;
                }
            }
        } else if key_tx.is_closed() {
            break;
        }
    });

    let mut terminal = ratatui::init();
    let mut app = App::new(banner);
    if let Some(g) = initial_goal {
        app.push_text("you › ", Style::default().fg(Color::Cyan), &g, 6);
        app.busy = true;
        let _ = in_tx.send(g);
    }
    let mut tick = tokio::time::interval(Duration::from_millis(120));
    let result = loop {
        if let Err(e) = terminal.draw(|f| draw(f, &mut app)) {
            break Err(anyhow::anyhow!(e));
        }
        tokio::select! {
            Some(ev) = ev_rx.recv() => app.apply(ev),
            Some(k) = key_rx.recv() => {
                if let CEvent::Key(key) = k {
                    if key.kind != KeyEventKind::Press { continue; }
                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                    match key.code {
                        KeyCode::Esc => app.should_quit = true,
                        KeyCode::Char('c') if ctrl => app.should_quit = true,
                        KeyCode::Char(c) if app.pending.is_some() && (c == 'y' || c == 'n' || c == 'Y' || c == 'N') => {
                            if let Some(p) = app.pending.take() {
                                let yes = c.eq_ignore_ascii_case(&'y');
                                let _ = p.reply.send(yes);
                                app.push(Line::from(Span::styled(format!("      {} {}", if yes { "allowed" } else { "denied" }, p.call), Style::default().dim())));
                            }
                        }
                        KeyCode::Enter => {
                            let g = app.input.trim().to_string();
                            if !g.is_empty() && app.pending.is_none() {
                                app.input.clear();
                                app.push_text("you › ", Style::default().fg(Color::Cyan), &g, 6);
                                app.busy = true;
                                let _ = in_tx.send(g);
                            }
                        }
                        KeyCode::Backspace => { app.input.pop(); }
                        KeyCode::Up => { app.follow = false; app.scroll = app.scroll.saturating_sub(1); }
                        KeyCode::Down => { app.scroll = app.scroll.saturating_add(1); }
                        KeyCode::PageUp => { app.follow = false; app.scroll = app.scroll.saturating_sub(10); }
                        KeyCode::PageDown => { app.scroll = app.scroll.saturating_add(10); }
                        KeyCode::End => { app.follow = true; app.scroll = u16::MAX; }
                        KeyCode::Char(c) if !ctrl => app.input.push(c),
                        _ => {}
                    }
                }
            }
            _ = tick.tick() => {}
        }
        if app.should_quit {
            break Ok(());
        }
    };
    ratatui::restore();
    result
}
