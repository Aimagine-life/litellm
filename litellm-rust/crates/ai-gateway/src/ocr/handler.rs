use litellm_core::auth::AuthorizeRequest;
use litellm_core::error::Error;
use litellm_core::http_utils::http_request;
use litellm_core::ocr::transformation::OcrResponseHandling;
use reqwest::Method;
use reqwest::header::CONTENT_TYPE;
use serde_json::Value;

use super::common_utils::{poll_document_intelligence, truncate_error_body};
use super::types::ProviderOcrRequest;
use crate::client::http_client;

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) async fn execute_ocr_provider_call(request: ProviderOcrRequest) -> Result<Value, Error> {
    let url = reqwest::Url::parse(&request.url)
        .map_err(|error| Error::InvalidRequest(format!("invalid OCR provider URL: {error}")))?;
    let serialized_body = serde_json::to_vec(&request.body)
        .map_err(|error| Error::InvalidRequest(format!("invalid OCR request body: {error}")))?;
    let authorized_headers = request
        .auth_session
        .authorize_primary(AuthorizeRequest {
            method: &Method::POST,
            url: &url,
            headers: request.upstream_headers,
            serialized_body: Some(&serialized_body),
        })
        .await?;
    let mut request_builder = http_client()
        .post(url)
        .header(CONTENT_TYPE, "application/json")
        .body(serialized_body);
    for (key, value) in &authorized_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = request.timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = http_request(request_builder)
        .await
        .map_err(|err| Error::Network(err.to_string()))?;

    let status = response.status();
    if request.config.response_handling() == OcrResponseHandling::AzureDocumentIntelligencePoll
        && status.as_u16() == 202
    {
        let operation_url = response
            .headers()
            .get("operation-location")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
            .ok_or_else(|| {
                Error::InvalidResponse(
                    "Azure Document Intelligence returned 202 but no Operation-Location header found"
                        .to_string(),
                )
            })?;
        let response_json =
            poll_document_intelligence(&operation_url, &request.auth_session, request.timeout)
                .await?;
        return Ok(request
            .config
            .transform_ocr_response_with_params(
                &request.model,
                response_json,
                &request.optional_params,
            )?
            .into_json());
    }

    let text = response
        .text()
        .await
        .map_err(|err| Error::Network(err.to_string()))?;

    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }

    let response_json: Value = serde_json::from_str(&text)
        .map_err(|err| Error::InvalidResponse(format!("invalid OCR response JSON: {err}")))?;

    Ok(request
        .config
        .transform_ocr_response_with_params(
            &request.model,
            response_json,
            &request.optional_params,
        )?
        .into_json())
}
