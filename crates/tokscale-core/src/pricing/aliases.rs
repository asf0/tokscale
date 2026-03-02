use once_cell::sync::Lazy;
use std::collections::HashMap;

static MODEL_ALIASES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut m = HashMap::new();
    m.insert("big-pickle", "glm-4.7");
    m.insert("big pickle", "glm-4.7");
    m.insert("bigpickle", "glm-4.7");
    m.insert("k2p5", "kimi-k2-thinking");
    m.insert("k2-p5", "kimi-k2-thinking");
    m.insert("kimi-k2.5-thinking", "kimi-k2-thinking");
    m.insert("kimi-for-coding", "kimi-k2.5");

    // synthetic.new model aliases (normalized from hf: prefix format)
    m.insert("deepseek-v3-0324", "deepseek/deepseek-chat");
    m.insert("deepseek-r1-0528", "deepseek/deepseek-reasoner");
    m.insert("deepseek-v3.2", "deepseek/deepseek-chat");
    m.insert("deepseek-v3", "deepseek/deepseek-chat");
    m.insert("kimi-k2.5-nvfp4", "kimi-k2.5");
    m.insert("kimi-k2-instruct-0905", "kimi-k2.5");
    m.insert("minimax-m2.1", "minimax/minimax-m1");
    m.insert("qwen3-235b-a22b-thinking-2507", "qwen/qwen3-235b-a22b");
    m.insert(
        "qwen3-coder-480b-a35b-instruct",
        "qwen/qwen3-coder-480b-a35b",
    );
    m.insert("gpt-oss-120b", "openai/gpt-4o");
    m.insert(
        "llama-3.3-70b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
    );
    m
});

pub fn resolve_alias(model_id: &str) -> Option<&'static str> {
    MODEL_ALIASES.get(model_id.to_lowercase().as_str()).copied()
}
