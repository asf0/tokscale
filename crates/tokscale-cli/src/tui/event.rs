use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, KeyEvent, MouseEvent};

pub enum Event {
    Tick,
    Key(KeyEvent),
    Mouse(MouseEvent),
    Resize(u16, u16),
}

pub struct EventHandler {
    rx: mpsc::Receiver<Event>,
}

impl EventHandler {
    pub fn new(tick_rate: Duration) -> Self {
        let (tx, rx) = mpsc::channel();
        let event_tx = tx.clone();

        thread::spawn(move || loop {
            if event::poll(tick_rate).unwrap_or(false) {
                match event::read() {
                    Ok(event::Event::Key(key)) => {
                        if event_tx.send(Event::Key(key)).is_err() {
                            break;
                        }
                    }
                    Ok(event::Event::Mouse(mouse)) => {
                        if event_tx.send(Event::Mouse(mouse)).is_err() {
                            break;
                        }
                    }
                    Ok(event::Event::Resize(w, h)) => {
                        if event_tx.send(Event::Resize(w, h)).is_err() {
                            break;
                        }
                    }
                    _ => {
                        // Prevent event starvation from FocusGained/FocusLost bursts
                        if event_tx.send(Event::Tick).is_err() {
                            break;
                        }
                    }
                }
            } else if event_tx.send(Event::Tick).is_err() {
                break;
            }
        });

        drop(tx);
        Self { rx }
    }

    pub fn next(&mut self) -> Result<Event> {
        Ok(self.rx.recv()?)
    }
}
