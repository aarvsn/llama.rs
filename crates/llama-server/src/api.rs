//! HTTP API: routes, request/response models and SSE streaming.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use llama_runtime::{Engine, Flow, GenerationConfig, StopReason};

/// Shared server state: the engine plus one serialized generation session.
///
/// Requests are serialized through a session mutex (one generation at a
/// time), matching the single-context behaviour of the llama.cpp server
/// without `--parallel`. Concurrent requests queue instead of corrupting the
/// KV cache.
pub struct AppState {
    engine: Engine,
    session: Mutex<llama_runtime::Session>,
    model_id: String,
    created: u64,
}

impl AppState {
    /// Creates the state from a loaded engine.
    #[must_use]
    pub fn new(engine: Engine) -> Self {
        let model_id = engine
            .info()
            .name
            .clone()
            .unwrap_or_else(|| "llama.rs model".to_string());
        Self {
            model_id,
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            session: Mutex::new(engine.create_session()),
            engine,
        }
    }

    /// The served model id.
    #[must_use]
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Builds the HTTP router.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Request / response models (OpenAI-compatible subset)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct CompletionRequest {
    /// The prompt (`/v1/completions`) or messages (`/v1/chat/completions`).
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    messages: Option<Vec<ChatMessage>>,
    /// Maximum tokens to generate.
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    /// Stop sequences (a single string or an array).
    #[serde(default)]
    stop: Vec<String>,
    /// Whether to stream tokens via SSE.
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    greedy: bool,
}

#[derive(Deserialize, Serialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

fn default_max_tokens() -> usize {
    128
}

fn default_temperature() -> f32 {
    0.8
}

impl CompletionRequest {
    fn to_config(&self) -> Result<GenerationConfig, (StatusCode, String)> {
        let mut cfg = GenerationConfig {
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_k: self.top_k.unwrap_or(40),
            top_p: self.top_p.unwrap_or(0.95),
            min_p: self.min_p.unwrap_or(0.05),
            repeat_penalty: self.repeat_penalty.unwrap_or(1.1),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            seed: self.seed,
            stop: self.stop.clone(),
            greedy: self.greedy || self.temperature == 0.0,
            parse_special: true,
            emit_special: false,
        };
        if cfg.temperature < 0.0 {
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                "temperature must be >= 0".into(),
            ));
        }
        if !(0.0..=1.0).contains(&cfg.top_p) {
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                "top_p must be in (0, 1]".into(),
            ));
        }
        if cfg.max_tokens == 0 {
            cfg.max_tokens = 1;
        }
        Ok(cfg)
    }

    fn validate_prompt(&self) -> Result<(), (StatusCode, String)> {
        if self.prompt.is_none() && self.messages.is_none() {
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                "provide 'prompt' or 'messages'".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<Choice>,
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    text: String,
    finish_reason: &'static str,
}

/// OpenAI-compatible usage block; the field names are the wire format.
#[derive(Serialize)]
#[allow(clippy::struct_field_names)]
struct Usage {
    prompt_tokens: u64,
    completion_tokens: u64,
    total_tokens: u64,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ErrorBody {
    error: ApiError,
}

#[derive(Serialize)]
struct ApiError {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
    code: &'static str,
}

fn api_error(status: StatusCode, message: impl Into<String>) -> Response {
    let body = ErrorBody {
        error: ApiError {
            message: message.into(),
            kind: "invalid_request_error",
            code: "",
        },
    };
    (status, Json(body)).into_response()
}

fn finish_reason_text(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Eos | StopReason::StopSequence | StopReason::Callback => "stop",
        StopReason::MaxTokens | StopReason::ContextFull => "length",
    }
}

fn response_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    format!("cmpl-{}-{}", std::process::id(), nanos)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

// axum's `Handler` trait is only implemented for async functions, so these
// handlers must stay `async` even without awaits.
#[allow(clippy::unused_async)]
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok", "engine": "llama.rs" }))
}

#[allow(clippy::unused_async)]
async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelList {
        object: "list",
        data: vec![ModelEntry {
            id: state.model_id.clone(),
            object: "model",
            created: state.created,
            owned_by: "llama.rs",
        }],
    })
}

async fn completions(
    State(state): State<Arc<AppState>>,
    body: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}")),
    };
    if let Err((status, msg)) = req.validate_prompt().and_then(|()| req.to_config()) {
        return api_error(status, msg);
    }
    let prompt = req.prompt.clone().unwrap_or_default();
    let config = req.to_config().expect("validated");

    if req.stream {
        stream_completion(&state, prompt, config)
    } else {
        let mut session = state.session.lock().await;
        match run_completion(&mut session, &prompt, &config) {
            Ok((text, usage, reason)) => {
                let resp = CompletionResponse {
                    id: response_id(),
                    object: "text_completion",
                    created: state.created,
                    model: state.model_id.clone(),
                    choices: vec![Choice {
                        index: 0,
                        text,
                        finish_reason: finish_reason_text(reason),
                    }],
                    usage,
                };
                Json(resp).into_response()
            }
            Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        }
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => return api_error(StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}")),
    };
    if let Err((status, msg)) = req.validate_prompt().and_then(|()| req.to_config()) {
        return api_error(status, msg);
    }
    let Some(messages) = req.messages.clone() else {
        return api_error(
            StatusCode::UNPROCESSABLE_ENTITY,
            "'messages' is required for chat completions",
        );
    };
    let config = req.to_config().expect("validated");

    // Render the conversation (or fall back to the plain template).
    let template = state
        .engine
        .chat_template()
        .unwrap_or(llama_tokenizer::ChatTemplate::Plain);
    let prompt = template.render(
        &messages
            .iter()
            .map(|m| llama_tokenizer::Message::new(&m.role, &m.content))
            .collect::<Vec<_>>(),
        state.engine.tokenizer().vocab(),
        true,
    );

    if req.stream {
        stream_chat(&state, prompt, config)
    } else {
        let mut session = state.session.lock().await;
        match run_completion(&mut session, &prompt, &config) {
            Ok((text, usage, reason)) => {
                let resp = ChatCompletionResponse {
                    id: response_id(),
                    object: "chat.completion",
                    created: state.created,
                    model: state.model_id.clone(),
                    choices: vec![ChatChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".into(),
                            content: text,
                        },
                        finish_reason: finish_reason_text(reason),
                    }],
                    usage,
                };
                Json(resp).into_response()
            }
            Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        }
    }
}

/// Runs a blocking generation on the shared session.
fn run_completion(
    session: &mut llama_runtime::Session,
    prompt: &str,
    config: &GenerationConfig,
) -> llama_core::Result<(String, Usage, StopReason)> {
    // Usage must describe *this* request: the session's cumulative
    // prefill counter also covers earlier requests on the shared context.
    let prompt_tokens = session
        .engine()
        .tokenizer()
        .encode(prompt, true, config.parse_special)
        .map(|t| t.len() as u64)?;
    session.set_prompt(prompt, config.parse_special)?;
    let out = session.generate(config, |_t, _p| Flow::Continue)?;
    let completion_tokens = out.tokens.len() as u64;
    let usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
    };
    Ok((out.text, usage, out.stop_reason))
}

/// Runs the generation loop on a blocking thread, streaming pieces over a
/// channel. Returns immediately; the response is fed by the receiver.
fn spawn_streaming_generation(
    state: &Arc<AppState>,
    prompt: String,
    config: GenerationConfig,
    wrap_piece: std::sync::Arc<dyn Fn(String) -> serde_json::Value + Send + Sync + 'static>,
) -> (
    tokio::sync::mpsc::Receiver<Result<Event, Infallible>>,
    Option<llama_core::LlamaError>,
) {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    // Prefill on a blocking thread; surface prefill errors through the
    // channel by returning them before spawning.
    let blocking_state = Arc::clone(state);
    let prompt_error = {
        // Try a quick synchronous validation of the prompt encoding first so
        // obvious failures return HTTP errors instead of empty streams.
        blocking_state
            .engine
            .tokenizer()
            .encode(&prompt, true, true)
            .err()
    };
    if let Some(e) = prompt_error {
        return (rx, Some(e));
    }

    tokio::task::spawn_blocking(move || {
        let mut session = blocking_state.session.blocking_lock();
        if let Err(e) = session.set_prompt(&prompt, config.parse_special) {
            let _ = tx.blocking_send(Ok(
                Event::default().data(serde_json::json!({"error": e.to_string()}).to_string())
            ));
            return;
        }
        let wrap = Arc::clone(&wrap_piece);
        let sender = std::sync::Arc::new(tx);
        let sender_done = Arc::clone(&sender);
        let _ = session.generate(&config, move |_token, piece: &str| {
            let payload = wrap(piece.to_string());
            let _ = sender.blocking_send(Ok(Event::default().data(payload.to_string())));
            Flow::Continue
        });
        let _ = sender_done.blocking_send(Ok(Event::default().data("[DONE]")));
    });

    (rx, None)
}

fn stream_completion(state: &Arc<AppState>, prompt: String, config: GenerationConfig) -> Response {
    let id = response_id();
    let wrap = std::sync::Arc::new(move |piece: String| {
        serde_json::json!({
            "id": id,
            "object": "text_completion",
            "choices": [{ "index": 0, "text": piece }],
        })
    });
    let (rx, err) = spawn_streaming_generation(state, prompt, config, wrap);
    if let Some(e) = err {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
    }
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn stream_chat(state: &Arc<AppState>, prompt: String, config: GenerationConfig) -> Response {
    let id = response_id();
    let model = state.model_id().to_string();
    let wrap = std::sync::Arc::new(move |piece: String| {
        serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "content": piece },
            }],
        })
    });
    let (rx, err) = spawn_streaming_generation(state, prompt, config, wrap);
    if let Some(e) = err {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
    }
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::header::CONTENT_TYPE;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn test_fixture_path() -> &'static std::path::Path {
        static FIXTURE_PATH: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
        FIXTURE_PATH.get_or_init(|| {
            let dir = std::env::temp_dir().join("llama-rs-server-tests");
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("server-fixture.gguf");
            llama_fixtures::write_llama_fixture(
                &path,
                &llama_fixtures::FixtureSpec {
                    quant: llama_fixtures::QuantScheme::F32,
                    ..llama_fixtures::FixtureSpec::default()
                },
            )
            .unwrap();
            path
        })
    }

    fn test_state() -> Arc<AppState> {
        let path = test_fixture_path();
        let engine = Engine::load(path).unwrap();
        Arc::new(AppState::new(engine))
    }

    #[tokio::test]
    async fn health_ok() {
        let app = router(test_state());
        let resp = app
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn models_listed() {
        let app = router(test_state());
        let resp = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["object"], "model");
    }

    #[tokio::test]
    async fn completion_generates_text() {
        let app = router(test_state());
        let req = Request::post("/v1/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::json!({
                    "prompt": "Once upon a time",
                    "max_tokens": 5,
                    "temperature": 0.0
                })
                .to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["choices"][0]["text"].is_string());
        assert!(json["usage"]["completion_tokens"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn bad_json_rejected() {
        let app = router(test_state());
        let req = Request::post("/v1/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from("{not json"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn missing_prompt_rejected() {
        let app = router(test_state());
        let req = Request::post("/v1/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::json!({"max_tokens": 4}).to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn chat_completion_uses_template() {
        let app = router(test_state());
        let req = Request::post("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(
                serde_json::json!({
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 4,
                    "temperature": 0.0
                })
                .to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    }
}
