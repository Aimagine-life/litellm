use std::time::Duration;

use reqwest::Method;
use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE, HeaderMap};
use serde_json::{Map, Value, json};

use super::common_utils::{convert_document_url_to_data_uri, poll_document_intelligence};
use super::types::PreparedOcrRequest;
use crate::auth::{AuthSession, AuthorizeRequest};
use crate::error::Error;
use crate::http_utils::{http_request, truncate_error_body};
use crate::ocr::transformation::{OcrProviderConfig, OcrResponseHandling};
use crate::providers::reducto::ocr::transformation::{
    build_upload_request, extract_document_source, extract_upload_file_id, upload_url,
};

fn public_response(
    config: &'static dyn OcrProviderConfig,
    model: &str,
    optional_params: &Map<String, Value>,
    response_json: Value,
) -> Result<Value, Error> {
    let mut response =
        config.transform_ocr_response_with_params(model, response_json, optional_params)?;
    response.provider_native_response = None;
    Ok(response.into_json())
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub(crate) async fn execute_ocr_provider_call(
    client: &reqwest::Client,
    request: PreparedOcrRequest,
) -> Result<Value, Error> {
    let PreparedOcrRequest {
        model,
        config,
        document,
        api_base,
        url,
        headers,
        auth_session,
        optional_params,
        requires_reducto_upload,
        timeout,
        max_document_download_bytes,
    } = request;
    let upstream_headers = request_headers(headers);
    let document = if config.requires_data_uri_document() {
        convert_document_url_to_data_uri(client, document, max_document_download_bytes).await?
    } else if requires_reducto_upload {
        upload_reducto_document(
            client,
            document,
            api_base.as_deref(),
            timeout,
            &auth_session,
            &upstream_headers,
        )
        .await?
    } else {
        document
    };
    let body = config
        .transform_ocr_request(&model, document, optional_params.clone())?
        .data;
    let serialized_body = serde_json::to_vec(&body)
        .map_err(|error| Error::InvalidRequest(format!("invalid OCR request body: {error}")))?;
    let authorized_headers = auth_session
        .authorize_primary(AuthorizeRequest {
            method: &Method::POST,
            url: &url,
            headers: upstream_headers,
            serialized_body: Some(&serialized_body),
        })
        .await?;
    let mut request_builder = client
        .post(url.clone())
        .header(CONTENT_TYPE, "application/json")
        .body(serialized_body);
    for (key, value) in &authorized_headers {
        request_builder = request_builder.header(key, value);
    }
    if let Some(duration) = timeout {
        request_builder = request_builder.timeout(duration);
    }

    let response = http_request(request_builder)
        .await
        .map_err(|err| Error::Network(err.to_string()))?;

    let status = response.status();
    if config.response_handling() == OcrResponseHandling::AzureDocumentIntelligencePoll
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
            poll_document_intelligence(client, &operation_url, &auth_session, timeout).await?;
        return public_response(config, &model, &optional_params, response_json);
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

    public_response(config, &model, &optional_params, response_json)
}

async fn upload_reducto_document(
    client: &reqwest::Client,
    document: Value,
    api_base: Option<&str>,
    timeout: Option<Duration>,
    auth_session: &AuthSession,
    upstream_headers: &HeaderMap,
) -> Result<Value, Error> {
    let source = extract_document_source(&document)?;
    let upload_url = reqwest::Url::parse(&upload_url(api_base))
        .map_err(|error| Error::InvalidRequest(format!("invalid Reducto upload URL: {error}")))?;
    let authorized_headers = auth_session
        .authorize_primary(AuthorizeRequest {
            method: &Method::POST,
            url: &upload_url,
            headers: upstream_headers.clone(),
            serialized_body: None,
        })
        .await?;
    let authorization = authorized_headers
        .get(reqwest::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            Error::Auth("Reducto upload requires an Authorization header".to_string())
        })?;
    let Some(upload) = build_upload_request(source, authorization, api_base) else {
        return Ok(document);
    };
    let part = reqwest::multipart::Part::bytes(upload.bytes)
        .file_name(upload.file_name)
        .mime_str(&upload.mime_type)
        .map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let form = reqwest::multipart::Form::new().part("file", part);
    let request_builder = authorized_headers.iter().fold(
        client.post(upload_url).multipart(form),
        |builder, (name, value)| builder.header(name, value),
    );
    let request_builder = match timeout {
        Some(duration) => request_builder.timeout(duration),
        None => request_builder,
    };
    let response = http_request(request_builder)
        .await
        .map_err(|error| Error::Network(error.to_string()))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| Error::Network(error.to_string()))?;
    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&body),
        });
    }
    let response_json: Value = serde_json::from_str(&body).map_err(|error| {
        Error::InvalidResponse(format!("invalid Reducto upload response JSON: {error}"))
    })?;
    let file_id = extract_upload_file_id(&response_json)?;
    Ok(json!({"type": "document_url", "document_url": file_id}))
}

fn request_headers(mut headers: HeaderMap) -> HeaderMap {
    headers.remove(CONTENT_TYPE);
    headers.remove(CONTENT_LENGTH);
    headers
}
