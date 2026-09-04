use std::future::Future;
use std::sync::Arc;

use litellm_core::Error;
use litellm_core::auth::TokenProvider;
use litellm_core::ocr::{OcrRequest, ocr as run_ocr};
use pyo3::prelude::*;
use serde_json::Value;

use crate::client::shared_http_client;
use crate::errors::ocr_error_to_pyerr;
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty};
use crate::python_token_provider::PythonTokenProvider;

fn prepare_ocr(
    inputs: OcrInputs,
) -> PyResult<impl Future<Output = Result<Value, Error>> + Send + 'static> {
    let external_token_provider = inputs
        .token_provider
        .map(|callable| Python::attach(|py| PythonTokenProvider::capture(py, callable)))
        .transpose()?
        .map(|provider| Arc::new(provider) as Arc<dyn TokenProvider>);
    let document = inputs.document;
    let options = RouteOptions::from_python(RouteOptionsInputs {
        model: inputs.model,
        api_key: inputs.api_key,
        api_base: inputs.api_base,
        custom_llm_provider: inputs.custom_llm_provider,
        extra_headers: inputs.extra_headers,
        timeout_seconds: inputs.timeout_seconds,
    })?;
    let optional_params = object_or_empty("optional_params", inputs.optional_params)?;

    Ok(async move {
        let client = shared_http_client().map_err(Error::Network)?;
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = options;
        run_ocr(
            &client,
            OcrRequest {
                model: &model,
                document,
                api_key: api_key.as_deref(),
                api_base: api_base.as_deref(),
                custom_llm_provider: custom_llm_provider.as_deref(),
                extra_headers,
                external_token_provider,
                optional_params,
                timeout,
                max_document_download_bytes: inputs.max_document_download_bytes,
            },
        )
        .await
    })
}

bridge_route! {
    sync = ocr,
    asynchronous = aocr,
    inputs = OcrInputs,
    required = {
        model: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        document: serde_json::Value,
        max_document_download_bytes: u64,
    },
    optional = {
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        extra_headers: Option<serde_json::Value>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        optional_params: Option<serde_json::Value>,
        timeout_seconds: Option<f64>,
        token_provider: Option<Py<PyAny>>,
    },
    prepare = prepare_ocr,
    errors = ocr_error_to_pyerr,
}
