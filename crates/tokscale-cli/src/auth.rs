use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn home_dir() -> Result<PathBuf> {
    dirs::home_dir().context("Could not determine home directory")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credentials {
    pub token: String,
    pub username: String,
    #[serde(rename = "avatarUrl", skip_serializing_if = "Option::is_none")]
    pub avatar_url: Option<String>,
    #[serde(rename = "createdAt")]
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
struct DeviceCodeResponse {
    #[serde(rename = "deviceCode")]
    device_code: String,
    #[serde(rename = "userCode")]
    user_code: String,
    #[serde(rename = "verificationUrl")]
    verification_url: String,
    #[serde(rename = "expiresIn")]
    #[allow(dead_code)]
    expires_in: u64,
    interval: u64,
}

#[derive(Debug, Deserialize)]
struct PollResponse {
    status: String,
    token: Option<String>,
    user: Option<UserInfo>,
    #[allow(dead_code)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct UserInfo {
    username: String,
    #[serde(rename = "avatarUrl")]
    avatar_url: Option<String>,
}

fn get_credentials_path() -> Result<PathBuf> {
    Ok(home_dir()?.join(".config/tokscale/credentials.json"))
}

fn ensure_config_dir() -> Result<()> {
    let config_dir = home_dir()?.join(".config/tokscale");

    if !config_dir.exists() {
        fs::create_dir_all(&config_dir)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&config_dir, fs::Permissions::from_mode(0o700))?;
        }
    }
    Ok(())
}

pub fn save_credentials(credentials: &Credentials) -> Result<()> {
    ensure_config_dir()?;
    let path = get_credentials_path()?;
    let json = serde_json::to_string_pretty(credentials)?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;

        let mut file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .mode(0o600)
            .open(&path)?;
        file.write_all(json.as_bytes())?;
    }

    #[cfg(not(unix))]
    {
        fs::write(&path, json)?;
    }

    Ok(())
}

pub fn load_credentials() -> Option<Credentials> {
    let path = get_credentials_path().ok()?;
    if !path.exists() {
        return None;
    }

    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

pub fn clear_credentials() -> Result<bool> {
    let path = get_credentials_path()?;
    if path.exists() {
        fs::remove_file(path)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

pub fn get_api_base_url() -> String {
    std::env::var("TOKSCALE_API_URL").unwrap_or_else(|_| "https://tokscale.ai".to_string())
}

fn get_device_name() -> String {
    let hostname = hostname::get()
        .ok()
        .and_then(|h| h.into_string().ok())
        .unwrap_or_else(|| "unknown".to_string());
    format!("CLI on {}", hostname)
}

fn open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(url).spawn();
    }

    #[cfg(target_os = "windows")]
    {
        let _ = std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn();
    }

    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("xdg-open").arg(url).spawn();
    }
}

pub async fn login() -> Result<()> {
    use colored::Colorize;

    if let Some(creds) = load_credentials() {
        println!(
            "\n  {}",
            format!("Already logged in as {}", creds.username.bold()).yellow()
        );
        println!(
            "{}",
            "  Run 'tokscale logout' to sign out first.\n".bright_black()
        );
        return Ok(());
    }

    let base_url = get_api_base_url();

    println!("\n  {}\n", "Tokscale - Login".cyan());
    println!("{}", "  Requesting authorization code...".bright_black());

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let device_code_response = client
        .post(format!("{}/api/auth/device", base_url))
        .json(&serde_json::json!({
            "deviceName": get_device_name()
        }))
        .send()
        .await?;

    if !device_code_response.status().is_success() {
        anyhow::bail!("Server returned {}", device_code_response.status());
    }

    let device_data: DeviceCodeResponse = device_code_response.json().await?;

    println!();
    println!("{}", "  Open this URL in your browser:".white());
    println!("{}", format!("  {}\n", device_data.verification_url).cyan());
    println!("{}", "  Enter this code:".white());
    println!(
        "{}\n",
        format!("  {}", device_data.user_code).green().bold()
    );

    open_browser(&device_data.verification_url);

    println!("{}", "  Waiting for authorization...".bright_black());

    let poll_interval = std::time::Duration::from_secs(device_data.interval);
    let max_attempts = 180;

    for attempt in 0..max_attempts {
        tokio::time::sleep(poll_interval).await;

        let poll_response = client
            .post(format!("{}/api/auth/device/poll", base_url))
            .json(&serde_json::json!({
                "deviceCode": device_data.device_code
            }))
            .send()
            .await;

        match poll_response {
            Ok(response) => {
                if let Ok(data) = response.json::<PollResponse>().await {
                    if data.status == "complete" {
                        if let (Some(token), Some(user)) = (data.token, data.user) {
                            let credentials = Credentials {
                                token,
                                username: user.username.clone(),
                                avatar_url: user.avatar_url,
                                created_at: chrono::Utc::now().to_rfc3339(),
                            };

                            save_credentials(&credentials)?;

                            println!(
                                "\n  {}",
                                format!("Success! Logged in as {}", user.username.bold()).green()
                            );
                            println!(
                                "{}",
                                "  You can now use 'tokscale submit' to share your usage.\n"
                                    .bright_black()
                            );
                            return Ok(());
                        }
                    }

                    if data.status == "expired" {
                        anyhow::bail!("Authorization code expired. Please try again.");
                    }

                    print!("{}", ".".bright_black());
                    use std::io::Write;
                    std::io::stdout().flush()?;
                }
            }
            Err(_) => {
                print!("{}", "!".red());
                use std::io::Write;
                std::io::stdout().flush()?;
            }
        }

        if attempt >= max_attempts - 1 {
            anyhow::bail!("Timeout: Authorization took too long. Please try again.");
        }
    }

    Ok(())
}

pub fn logout() -> Result<()> {
    use colored::Colorize;

    let credentials = load_credentials();

    let Some(creds) = credentials else {
        println!("\n  {}\n", "Not logged in.".yellow());
        return Ok(());
    };

    let username = creds.username;
    let cleared = clear_credentials()?;

    if cleared {
        println!(
            "\n  {}\n",
            format!("Logged out from {}", username.bold()).green()
        );
    } else {
        anyhow::bail!("Failed to clear credentials.");
    }

    Ok(())
}

pub fn whoami() -> Result<()> {
    use colored::Colorize;

    let Some(creds) = load_credentials() else {
        println!("\n  {}", "Not logged in.".yellow());
        println!(
            "{}",
            "  Run 'tokscale login' to authenticate.\n".bright_black()
        );
        return Ok(());
    };

    println!("\n  {}\n", "Tokscale - Account Info".cyan());
    println!(
        "{}",
        format!("  Username:  {}", creds.username.bold()).white()
    );

    if let Ok(created) = chrono::DateTime::parse_from_rfc3339(&creds.created_at) {
        println!(
            "{}",
            format!("  Logged in: {}", created.format("%Y-%m-%d")).bright_black()
        );
    }

    println!();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::env;
    use tempfile::TempDir;

    #[test]
    #[serial]
    fn test_get_api_base_url_default() {
        unsafe {
            env::remove_var("TOKSCALE_API_URL");
        }
        assert_eq!(get_api_base_url(), "https://tokscale.ai");
    }

    #[test]
    #[serial]
    fn test_get_api_base_url_custom() {
        unsafe {
            env::set_var("TOKSCALE_API_URL", "https://custom.api.url");
        }
        assert_eq!(get_api_base_url(), "https://custom.api.url");
        unsafe {
            env::remove_var("TOKSCALE_API_URL");
        }
    }

    #[test]
    fn test_credentials_serialization() {
        let creds = Credentials {
            token: "test_token_123".to_string(),
            username: "testuser".to_string(),
            avatar_url: Some("https://example.com/avatar.png".to_string()),
            created_at: "2024-01-01T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&creds).unwrap();
        let deserialized: Credentials = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.token, creds.token);
        assert_eq!(deserialized.username, creds.username);
        assert_eq!(deserialized.avatar_url, creds.avatar_url);
        assert_eq!(deserialized.created_at, creds.created_at);
    }

    #[test]
    fn test_credentials_serialization_without_avatar() {
        let creds = Credentials {
            token: "test_token_456".to_string(),
            username: "testuser2".to_string(),
            avatar_url: None,
            created_at: "2024-01-02T00:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&creds).unwrap();
        let deserialized: Credentials = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.token, creds.token);
        assert_eq!(deserialized.username, creds.username);
        assert_eq!(deserialized.avatar_url, None);
        assert_eq!(deserialized.created_at, creds.created_at);

        assert!(!json.contains("avatarUrl"));
    }

    #[test]
    #[serial]
    fn test_get_credentials_path() {
        let temp_dir = TempDir::new().unwrap();
        unsafe {
            env::set_var("HOME", temp_dir.path());
        }

        let path = get_credentials_path().unwrap();
        let expected = temp_dir.path().join(".config/tokscale/credentials.json");

        assert_eq!(path, expected);

        unsafe {
            env::remove_var("HOME");
        }
    }

    #[test]
    #[serial]
    fn test_save_credentials() {
        let temp_dir = TempDir::new().unwrap();
        unsafe {
            env::set_var("HOME", temp_dir.path());
        }

        let creds = Credentials {
            token: "save_test_token".to_string(),
            username: "saveuser".to_string(),
            avatar_url: Some("https://example.com/save.png".to_string()),
            created_at: "2024-01-03T00:00:00Z".to_string(),
        };

        save_credentials(&creds).unwrap();

        let path = get_credentials_path().unwrap();
        assert!(path.exists());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(&path).unwrap();
            let permissions = metadata.permissions();
            assert_eq!(permissions.mode() & 0o777, 0o600);
        }

        let content = fs::read_to_string(&path).unwrap();
        let loaded: Credentials = serde_json::from_str(&content).unwrap();
        assert_eq!(loaded.token, creds.token);
        assert_eq!(loaded.username, creds.username);

        unsafe {
            env::remove_var("HOME");
        }
    }

    #[test]
    #[serial]
    fn test_load_credentials() {
        let temp_dir = TempDir::new().unwrap();
        unsafe {
            env::set_var("HOME", temp_dir.path());
        }

        let creds = Credentials {
            token: "load_test_token".to_string(),
            username: "loaduser".to_string(),
            avatar_url: None,
            created_at: "2024-01-04T00:00:00Z".to_string(),
        };

        save_credentials(&creds).unwrap();

        let loaded = load_credentials().unwrap();

        assert_eq!(loaded.token, creds.token);
        assert_eq!(loaded.username, creds.username);
        assert_eq!(loaded.avatar_url, creds.avatar_url);
        assert_eq!(loaded.created_at, creds.created_at);

        unsafe {
            env::remove_var("HOME");
        }
    }

    #[test]
    #[serial]
    fn test_load_credentials_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        unsafe {
            env::set_var("HOME", temp_dir.path());
        }

        let loaded = load_credentials();
        assert!(loaded.is_none());

        unsafe {
            env::remove_var("HOME");
        }
    }

    #[test]
    #[serial]
    fn test_clear_credentials() {
        let temp_dir = TempDir::new().unwrap();
        unsafe {
            env::set_var("HOME", temp_dir.path());
        }

        let creds = Credentials {
            token: "clear_test_token".to_string(),
            username: "clearuser".to_string(),
            avatar_url: None,
            created_at: "2024-01-05T00:00:00Z".to_string(),
        };

        save_credentials(&creds).unwrap();
        let path = get_credentials_path().unwrap();
        assert!(path.exists());

        let cleared = clear_credentials().unwrap();
        assert!(cleared);
        assert!(!path.exists());

        unsafe {
            env::remove_var("HOME");
        }
    }

    #[test]
    #[serial]
    fn test_clear_credentials_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        unsafe {
            env::set_var("HOME", temp_dir.path());
        }

        let cleared = clear_credentials().unwrap();
        assert!(!cleared);

        unsafe {
            env::remove_var("HOME");
        }
    }

    #[test]
    #[serial]
    fn test_ensure_config_dir() {
        let temp_dir = TempDir::new().unwrap();
        unsafe {
            env::set_var("HOME", temp_dir.path());
        }

        let config_dir = temp_dir.path().join(".config/tokscale");
        assert!(!config_dir.exists());

        ensure_config_dir().unwrap();

        assert!(config_dir.exists());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(&config_dir).unwrap();
            let permissions = metadata.permissions();
            assert_eq!(permissions.mode() & 0o777, 0o700);
        }

        unsafe {
            env::remove_var("HOME");
        }
    }

    #[test]
    #[serial]
    fn test_save_and_load_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        unsafe {
            env::set_var("HOME", temp_dir.path());
        }

        let original = Credentials {
            token: "roundtrip_token".to_string(),
            username: "roundtripuser".to_string(),
            avatar_url: Some("https://example.com/roundtrip.png".to_string()),
            created_at: "2024-01-06T12:34:56Z".to_string(),
        };

        save_credentials(&original).unwrap();
        let loaded = load_credentials().unwrap();

        assert_eq!(loaded.token, original.token);
        assert_eq!(loaded.username, original.username);
        assert_eq!(loaded.avatar_url, original.avatar_url);
        assert_eq!(loaded.created_at, original.created_at);

        unsafe {
            env::remove_var("HOME");
        }
    }
}
