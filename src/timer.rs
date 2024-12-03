use std::time::{Duration, Instant};

/// Timer.
#[derive(Clone)]
pub struct Timer {
    start: Instant,
    elapsed_time: Duration,
    time_limit: Option<Duration>,
}

impl Default for Timer {
    fn default() -> Self {
        Self {
            start: Instant::now(),
            elapsed_time: Duration::from_secs(0),
            time_limit: None,
        }
    }
}

impl Timer {
    /// Returns a time keeper with the given time limit.
    pub fn with_time_limit(time_limit: f64) -> Self {
        Self {
            start: Instant::now(),
            elapsed_time: Duration::from_secs(0),
            time_limit: Some(Duration::from_secs_f64(time_limit)),
        }
    }

    /// Starts the time keeper.
    pub fn start(&mut self) {
        self.start = Instant::now()
    }

    /// Stops the time keeper.
    pub fn stop(&mut self) {
        self.elapsed_time += Instant::now() - self.start;
    }

    /// Returns the elapsed time.
    pub fn get_elapsed_time(&self) -> f64 {
        (self.elapsed_time + (Instant::now() - self.start)).as_secs_f64()
    }

    /// Returns the remaining time.
    pub fn get_remaining_time_limit(&self) -> Option<f64> {
        let elapsed_time = self.elapsed_time + (Instant::now() - self.start);
        self.time_limit.map(|time_limit| {
            if elapsed_time > time_limit {
                0.0
            } else {
                (time_limit - elapsed_time).as_secs_f64()
            }
        })
    }

    /// Returns whether the time limit is reached.
    pub fn check_time_limit(&self) -> bool {
        if let Some(remaining) = self.get_remaining_time_limit() {
            remaining <= 0.0
        } else {
            false
        }
    }
}
