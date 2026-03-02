use super::cache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const CACHE_FILENAME: &str = "pricing-litellm.json";
const PRICING_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";
const MAX_RETRIES: u32 = 3;
const INITIAL_BACKOFF_MS: u64 = 200;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelPricing {
    pub input_cost_per_token: Option<f64>,
    pub output_cost_per_token: Option<f64>,
    pub cache_creation_input_token_cost: Option<f64>,
    pub cache_read_input_token_cost: Option<f64>,
}

pub type PricingDataset = HashMap<String, ModelPricing>;

pub fn load_cached() -> Option<PricingDataset> {
    cache::load_cache(CACHE_FILENAME)
}

pub async fn fetch() -> Result<PricingDataset, reqwest::Error> {
    if let Some(cached) = load_cached() {
        return Ok(cached);
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()?;

    let mut last_error: Option<reqwest::Error> = None;

    for attempt in 0..MAX_RETRIES {
        match client.get(PRICING_URL).send().await {
            Ok(response) => {
                let status = response.status();

                if status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS {
                    eprintln!(
                        "[tokscale] LiteLLM HTTP {} (attempt {}/{})",
                        status,
                        attempt + 1,
                        MAX_RETRIES
                    );
                    let _ = response.bytes().await;
                    if attempt < MAX_RETRIES - 1 {
                        tokio::time::sleep(std::time::Duration::from_millis(
                            INITIAL_BACKOFF_MS * (1 << attempt),
                        ))
                        .await;
                    }
                    continue;
                }

                if !status.is_success() {
                    eprintln!("[tokscale] LiteLLM HTTP {}", status);
                    return Err(response.error_for_status().unwrap_err());
                }

                match response.json::<PricingDataset>().await {
                    Ok(data) => {
                        if let Err(e) = cache::save_cache(CACHE_FILENAME, &data) {
                            eprintln!(
                                "[tokscale] Warning: Failed to cache LiteLLM pricing at {}: {}",
                                cache::get_cache_path(CACHE_FILENAME).display(),
                                e
                            );
                        }
                        return Ok(data);
                    }
                    Err(e) => {
                        eprintln!("[tokscale] LiteLLM JSON parse failed: {}", e);
                        return Err(e);
                    }
                }
            }
            Err(e) => {
                eprintln!(
                    "[tokscale] LiteLLM network error (attempt {}/{}): {}",
                    attempt + 1,
                    MAX_RETRIES,
                    e
                );
                last_error = Some(e);
                if attempt < MAX_RETRIES - 1 {
                    tokio::time::sleep(std::time::Duration::from_millis(
                        INITIAL_BACKOFF_MS * (1 << attempt),
                    ))
                    .await;
                }
            }
        }
    }

    Err(last_error.expect("should have error after retries"))
}
