//! Kilocode session parser
//!
//! Parses messages from: ~/.local/share/kilo/storage/message/*/*.json

use super::{normalize_agent_name, UnifiedMessage};
use crate::TokenBreakdown;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct KiloMessage {
    pub id: Option<String>,
    #[serde(rename = "sessionID", default)]
    pub session_id: Option<String>,
    pub role: String,
    #[serde(rename = "modelID")]
    pub model_id: Option<String>,
    #[serde(rename = "providerID")]
    pub provider_id: Option<String>,
    pub cost: Option<f64>,
    pub tokens: Option<KiloTokens>,
    pub time: KiloTime,
    pub agent: Option<String>,
    pub mode: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct KiloTokens {
    pub input: i64,
    pub output: i64,
    pub reasoning: Option<i64>,
    pub cache: KiloCache,
}

#[derive(Debug, Deserialize)]
pub struct KiloCache {
    pub read: i64,
    pub write: i64,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct KiloTime {
    pub created: f64,
    pub completed: Option<f64>,
}

pub fn parse_kilo_file(path: &Path) -> Option<UnifiedMessage> {
    let data = std::fs::read(path).ok()?;
    let mut bytes = data;

    let msg: KiloMessage = simd_json::from_slice(&mut bytes).ok()?;

    if msg.role != "assistant" {
        return None;
    }

    let tokens = msg.tokens?;
    let model_id = msg.model_id?;
    let agent_or_mode = msg.mode.or(msg.agent);
    let agent = agent_or_mode.map(|a| normalize_agent_name(&a));

    let session_id = msg.session_id.unwrap_or_else(|| "unknown".to_string());

    let dedup_key = msg.id.or_else(|| {
        path.file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
    });

    let mut unified = UnifiedMessage::new_with_agent(
        "kilo",
        model_id,
        msg.provider_id.unwrap_or_else(|| "kilo".to_string()),
        session_id,
        msg.time.created as i64,
        TokenBreakdown {
            input: tokens.input.max(0),
            output: tokens.output.max(0),
            cache_read: tokens.cache.read.max(0),
            cache_write: tokens.cache.write.max(0),
            reasoning: tokens.reasoning.unwrap_or(0).max(0),
        },
        msg.cost.unwrap_or(0.0).max(0.0),
        agent,
    );
    unified.dedup_key = dedup_key;
    Some(unified)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_parse_kilo_structure() {
        let json = r#"{
            "id": "msg_123",
            "sessionID": "ses_456",
            "role": "assistant",
            "modelID": "z-ai/glm-5:free",
            "providerID": "kilo",
            "cost": 0.05,
            "tokens": {
                "total": 1000,
                "input": 800,
                "output": 200,
                "reasoning": 50,
                "cache": { "read": 100, "write": 25 }
            },
            "time": { "created": 1700000000000.0 }
        }"#;

        let mut bytes = json.as_bytes().to_vec();
        let msg: KiloMessage = simd_json::from_slice(&mut bytes).unwrap();

        assert_eq!(msg.model_id, Some("z-ai/glm-5:free".to_string()));
        assert_eq!(msg.tokens.unwrap().input, 800);
        assert_eq!(msg.agent, None);
    }

    #[test]
    fn test_parse_kilo_with_agent() {
        let json = r#"{
            "id": "msg_123",
            "sessionID": "ses_456",
            "role": "assistant",
            "modelID": "z-ai/glm-5:free",
            "providerID": "kilo",
            "agent": "explore",
            "cost": 0.05,
            "tokens": {
                "total": 1000,
                "input": 800,
                "output": 200,
                "reasoning": 50,
                "cache": { "read": 100, "write": 25 }
            },
            "time": { "created": 1700000000000.0 }
        }"#;

        let mut bytes = json.as_bytes().to_vec();
        let msg: KiloMessage = simd_json::from_slice(&mut bytes).unwrap();

        assert_eq!(msg.agent, Some("explore".to_string()));
    }

    #[test]
    fn test_negative_values_clamped_to_zero() {
        let json = r#"{
            "id": "msg_negative",
            "sessionID": "ses_negative",
            "role": "assistant",
            "modelID": "z-ai/glm-5:free",
            "providerID": "kilo",
            "cost": -0.05,
            "tokens": {
                "total": -100,
                "input": -100,
                "output": -50,
                "reasoning": -25,
                "cache": { "read": -200, "write": -10 }
            },
            "time": { "created": 1700000000000.0 }
        }"#;

        let mut temp_file = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        temp_file.write_all(json.as_bytes()).unwrap();

        let result = parse_kilo_file(temp_file.path());
        assert!(result.is_some(), "Should parse file with negative values");

        let msg = result.unwrap();
        assert_eq!(msg.tokens.input, 0, "Negative input should be clamped to 0");
        assert_eq!(msg.tokens.output, 0, "Negative output should be clamped to 0");
        assert_eq!(msg.tokens.cache_read, 0, "Negative cache_read should be clamped to 0");
        assert_eq!(msg.tokens.cache_write, 0, "Negative cache_write should be clamped to 0");
        assert_eq!(msg.tokens.reasoning, 0, "Negative reasoning should be clamped to 0");
        assert!(msg.cost >= 0.0, "Negative cost should be clamped to 0.0, got {}", msg.cost);
    }

    #[test]
    fn test_dedup_key_from_json_message_id() {
        let json = r#"{
            "id": "msg_dedup_001",
            "sessionID": "ses_001",
            "role": "assistant",
            "modelID": "z-ai/glm-5:free",
            "providerID": "kilo",
            "cost": 0.01,
            "tokens": {
                "total": 150,
                "input": 100,
                "output": 50,
                "reasoning": 0,
                "cache": { "read": 0, "write": 0 }
            },
            "time": { "created": 1700000000000.0 }
        }"#;

        let mut temp_file = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        temp_file.write_all(json.as_bytes()).unwrap();

        let msg = parse_kilo_file(temp_file.path()).expect("Should parse");
        assert_eq!(
            msg.dedup_key,
            Some("msg_dedup_001".to_string()),
            "dedup_key should use msg.id from JSON"
        );
    }

    #[test]
    fn test_dedup_key_falls_back_to_file_stem() {
        let json = r#"{
            "sessionID": "ses_001",
            "role": "assistant",
            "modelID": "z-ai/glm-5:free",
            "providerID": "kilo",
            "cost": 0.01,
            "tokens": {
                "total": 150,
                "input": 100,
                "output": 50,
                "reasoning": 0,
                "cache": { "read": 0, "write": 0 }
            },
            "time": { "created": 1700000000000.0 }
        }"#;

        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("msg_fallback_999.json");
        std::fs::write(&file_path, json).unwrap();

        let msg = parse_kilo_file(&file_path).expect("Should parse");
        assert_eq!(
            msg.dedup_key,
            Some("msg_fallback_999".to_string()),
            "dedup_key should fall back to file stem when id is missing"
        );
    }

    #[test]
    fn test_non_assistant_messages_skipped() {
        let json = r#"{
            "id": "msg_user_001",
            "sessionID": "ses_001",
            "role": "user",
            "modelID": "z-ai/glm-5:free",
            "providerID": "kilo",
            "tokens": {
                "total": 150,
                "input": 100,
                "output": 50,
                "reasoning": 0,
                "cache": { "read": 0, "write": 0 }
            },
            "time": { "created": 1700000000000.0 }
        }"#;

        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("msg_user_001.json");
        std::fs::write(&file_path, json).unwrap();

        let result = parse_kilo_file(&file_path);
        assert!(result.is_none(), "User messages should be skipped");
    }

    #[test]
    fn test_source_is_kilo() {
        let json = r#"{
            "id": "msg_source_test",
            "sessionID": "ses_001",
            "role": "assistant",
            "modelID": "z-ai/glm-5:free",
            "providerID": "kilo",
            "cost": 0.01,
            "tokens": {
                "total": 150,
                "input": 100,
                "output": 50,
                "reasoning": 0,
                "cache": { "read": 0, "write": 0 }
            },
            "time": { "created": 1700000000000.0 }
        }"#;

        let mut temp_file = tempfile::Builder::new().suffix(".json").tempfile().unwrap();
        temp_file.write_all(json.as_bytes()).unwrap();

        let msg = parse_kilo_file(temp_file.path()).expect("Should parse");
        assert_eq!(msg.source, "kilo");
    }
}
