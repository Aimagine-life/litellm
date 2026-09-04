use std::sync::Arc;
use std::time::Duration;

use reqwest::Url;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};

use super::transformation::{OcrProviderConfig, OcrResponseHandling};
use crate::Error;
use crate::auth::{
    AuthError, AuthErrorKind, AuthHeaderKind, AuthInput, AuthRuntime, AuthSession, Authenticator,
    BearerTokenAuthorizer, ReplayPolicy, SecretString, StaticHeaderAuthorizer, SystemClock,
    TokenCache, TokenProvider,
};

pub struct OcrAuthenticator {
    kind: AuthHeaderKind,
    replay: ReplayPolicy,
}

impl OcrAuthenticator {
    pub fn new(config: &'static dyn OcrProviderConfig) -> Self {
        let replay = match config.response_handling() {
            OcrResponseHandling::AzureDocumentIntelligencePoll => ReplayPolicy::SameOrigin,
            OcrResponseHandling::Json => ReplayPolicy::Never,
        };
        Self {
            kind: config.auth_header_kind(),
            replay,
        }
    }

    fn conflicts(&self) -> Result<Vec<HeaderName>, AuthError> {
        let direct = HeaderName::from_bytes(self.kind.header_name().as_bytes()).map_err(|_| {
            AuthError::new(
                AuthErrorKind::InvalidConfiguration,
                "invalid_auth_header_name",
                "provider configured an invalid authentication header name",
            )
        })?;
        Ok(if direct == AUTHORIZATION {
            vec![AUTHORIZATION]
        } else {
            vec![direct, AUTHORIZATION]
        })
    }

    fn forwarded_authorizer(
        &self,
        headers: &HeaderMap,
        conflicts: Vec<HeaderName>,
    ) -> Result<Option<StaticHeaderAuthorizer>, AuthError> {
        let direct = HeaderName::from_bytes(self.kind.header_name().as_bytes()).map_err(|_| {
            AuthError::new(
                AuthErrorKind::InvalidConfiguration,
                "invalid_auth_header_name",
                "provider configured an invalid authentication header name",
            )
        })?;
        let selected = headers
            .get(&direct)
            .map(|value| (direct, value))
            .or_else(|| {
                headers
                    .get(AUTHORIZATION)
                    .map(|value| (AUTHORIZATION, value))
            });
        let Some((name, value)) = selected else {
            return Ok(None);
        };
        let value = value.to_str().map_err(|_| {
            AuthError::new(
                AuthErrorKind::InvalidHeader,
                "invalid_forwarded_credential",
                "forwarded credential is not a valid HTTP header value",
            )
        })?;
        if value.trim().is_empty() {
            return Err(AuthError::new(
                AuthErrorKind::InvalidConfiguration,
                "blank_explicit_credential",
                "explicit authentication credentials cannot be blank",
            ));
        }
        Ok(Some(StaticHeaderAuthorizer::new(
            name,
            SecretString::new(value),
            conflicts,
        )))
    }
}

#[async_trait::async_trait]
impl Authenticator for OcrAuthenticator {
    async fn authenticate(
        &self,
        input: AuthInput<'_>,
        runtime: &AuthRuntime,
    ) -> Result<AuthSession, AuthError> {
        let conflicts = self.conflicts()?;
        if let Some(authorizer) =
            self.forwarded_authorizer(input.forwarded_headers, conflicts.clone())?
        {
            return Ok(AuthSession::new(Arc::new(authorizer), self.replay));
        }
        if let Some(api_key) = input.explicit_api_key {
            if api_key.trim().is_empty() {
                return Err(AuthError::new(
                    AuthErrorKind::InvalidConfiguration,
                    "blank_explicit_credential",
                    "explicit authentication credentials cannot be blank",
                ));
            }
            let name =
                HeaderName::from_bytes(self.kind.header_name().as_bytes()).map_err(|_| {
                    AuthError::new(
                        AuthErrorKind::InvalidConfiguration,
                        "invalid_auth_header_name",
                        "provider configured an invalid authentication header name",
                    )
                })?;
            let value = self.kind.header_value(&SecretString::new(api_key.trim()));
            return Ok(AuthSession::new(
                Arc::new(StaticHeaderAuthorizer::new(name, value, conflicts)),
                self.replay,
            ));
        }
        let provider = input.external_token_provider.ok_or_else(|| {
            AuthError::new(
                AuthErrorKind::MissingCredential,
                "missing_credential",
                "no provider credential was configured",
            )
        })?;
        Ok(AuthSession::new(
            Arc::new(BearerTokenAuthorizer::new(
                provider,
                runtime.clock.clone(),
                conflicts,
            )),
            self.replay,
        ))
    }
}

pub fn header_map(headers: Vec<(String, String)>) -> Result<HeaderMap, Error> {
    headers
        .into_iter()
        .try_fold(HeaderMap::new(), |mut result, (name, value)| {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(|_| Error::InvalidRequest(format!("invalid OCR header name: {name}")))?;
            let value = HeaderValue::from_str(&value)
                .map_err(|_| Error::InvalidRequest("invalid OCR header value".to_string()))?;
            result.insert(name, value);
            Ok(result)
        })
}

pub fn runtime() -> AuthRuntime {
    let clock = Arc::new(SystemClock);
    AuthRuntime {
        environment: Arc::new(|name: &str| std::env::var(name).ok()),
        clock: clock.clone(),
        tokens: Arc::new(TokenCache::new(Duration::from_secs(30), clock)),
    }
}

pub async fn auth_session(
    config: &'static dyn OcrProviderConfig,
    api_key: Option<&str>,
    headers: &HeaderMap,
    external_token_provider: Option<Arc<dyn TokenProvider>>,
    runtime: &AuthRuntime,
    url: &Url,
) -> Result<AuthSession, Error> {
    OcrAuthenticator::new(config)
        .authenticate(
            AuthInput {
                explicit_api_key: api_key,
                forwarded_headers: headers,
                external_token_provider,
            },
            runtime,
        )
        .await
        .map(|session| session.bind(url))
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use reqwest::Method;

    use super::*;
    use crate::auth::{AuthorizeRequest, TokenCredential};
    use crate::providers::azure_ai::ocr::transformation::AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG;
    use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;

    struct CountingProvider(AtomicUsize);

    #[async_trait]
    impl TokenProvider for CountingProvider {
        async fn token(&self) -> Result<TokenCredential, AuthError> {
            let call = self.0.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(TokenCredential::NoStore(SecretString::new(format!(
                "token-{call}"
            ))))
        }
    }

    #[tokio::test]
    async fn python_provider_is_resolved_only_when_authorizing() {
        let provider = Arc::new(CountingProvider(AtomicUsize::new(0)));
        let url = Url::parse("https://api.mistral.ai/v1/ocr").unwrap();
        let session = auth_session(
            &MISTRAL_OCR_CONFIG,
            None,
            &HeaderMap::new(),
            Some(provider.clone()),
            &runtime(),
            &url,
        )
        .await
        .unwrap();
        assert_eq!(provider.0.load(Ordering::SeqCst), 0);

        let headers = session
            .authorize_primary(AuthorizeRequest {
                method: &Method::POST,
                url: &url,
                headers: HeaderMap::new(),
                serialized_body: Some(b"{}"),
            })
            .await
            .unwrap();
        assert_eq!(headers.get(AUTHORIZATION).unwrap(), "Bearer token-1");
        assert_eq!(provider.0.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn blank_explicit_key_does_not_fall_back_to_provider() {
        let provider = Arc::new(CountingProvider(AtomicUsize::new(0)));
        let url = Url::parse("https://api.mistral.ai/v1/ocr").unwrap();
        let error = auth_session(
            &MISTRAL_OCR_CONFIG,
            Some("  "),
            &HeaderMap::new(),
            Some(provider.clone()),
            &runtime(),
            &url,
        )
        .await
        .err()
        .unwrap();

        assert!(error.to_string().contains("cannot be blank"));
        assert_eq!(provider.0.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn document_intelligence_replays_only_auth_to_same_origin() {
        let url = Url::parse("https://resource.example/documentintelligence/analyze").unwrap();
        let headers = HeaderMap::from_iter([
            (
                HeaderName::from_static("ocp-apim-subscription-key"),
                HeaderValue::from_static("subscription-key"),
            ),
            (
                HeaderName::from_static("x-request-only"),
                HeaderValue::from_static("value"),
            ),
        ]);
        let session = auth_session(
            &AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG,
            None,
            &headers,
            None,
            &runtime(),
            &url,
        )
        .await
        .unwrap();
        let poll_url = Url::parse("https://resource.example/operations/1").unwrap();
        let poll_headers = session
            .authorize_follow_up(AuthorizeRequest {
                method: &Method::GET,
                url: &poll_url,
                headers: HeaderMap::new(),
                serialized_body: None,
            })
            .await
            .unwrap();
        assert_eq!(
            poll_headers.get("ocp-apim-subscription-key").unwrap(),
            "subscription-key"
        );
        assert!(!poll_headers.contains_key("x-request-only"));

        let cross_origin = Url::parse("https://attacker.example/operations/1").unwrap();
        assert!(
            session
                .authorize_follow_up(AuthorizeRequest {
                    method: &Method::GET,
                    url: &cross_origin,
                    headers: HeaderMap::new(),
                    serialized_body: None,
                })
                .await
                .is_err()
        );
    }
}
