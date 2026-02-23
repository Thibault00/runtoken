use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::{Tokenizer, TokenizerRegistry};
use std::path::Path;
use std::sync::Mutex;

/// Thread-safe wrapper around the Rust Tokenizer for Python.
#[pyclass]
struct PyTokenizer {
    inner: Mutex<Tokenizer>,
}

#[pymethods]
impl PyTokenizer {
    /// Create a new tokenizer for a given encoding.
    ///
    /// Args:
    ///     encoding_name: One of "cl100k_base", "o200k_base", "p50k_base"
    ///     vocab_path: Optional path to a .tiktoken vocab file (uses bundled if omitted)
    #[new]
    #[pyo3(signature = (encoding_name, vocab_path=None))]
    fn new(encoding_name: &str, vocab_path: Option<&str>) -> PyResult<Self> {
        let inner = if let Some(path) = vocab_path {
            Tokenizer::from_file(encoding_name, Path::new(path))
        } else {
            Tokenizer::new(encoding_name)
        };
        inner
            .map(|t| PyTokenizer { inner: Mutex::new(t) })
            .map_err(|e| PyValueError::new_err(format!("Failed to load tokenizer: {}", e)))
    }

    /// Encode text to a list of token IDs.
    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        let tok = self.inner.lock().map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;
        Ok(tok.encode(text))
    }

    /// Count tokens in text.
    fn count(&self, text: &str) -> PyResult<usize> {
        let tok = self.inner.lock().map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;
        Ok(tok.count(text))
    }

    /// Decode token IDs back to text.
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        let tok = self.inner.lock().map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;
        Ok(tok.decode(&tokens))
    }

    /// Get the encoding name.
    #[getter]
    fn encoding_name(&self) -> PyResult<String> {
        let tok = self.inner.lock().map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;
        Ok(tok.encoding_name.clone())
    }

    /// Get vocab size.
    #[getter]
    fn vocab_size(&self) -> PyResult<usize> {
        let tok = self.inner.lock().map_err(|e| PyValueError::new_err(format!("Lock error: {}", e)))?;
        Ok(tok.vocab.encoder.len())
    }
}

/// Get a tokenizer for an encoding name.
///
/// >>> import runtoken
/// >>> enc = runtoken.get_encoding("cl100k_base")
/// >>> enc.encode("Hello, world!")
/// [9906, 11, 1917, 0]
#[pyfunction]
fn get_encoding(encoding_name: &str) -> PyResult<PyTokenizer> {
    PyTokenizer::new(encoding_name, None)
}

/// Get a tokenizer for a model name (maps model -> encoding automatically).
///
/// >>> import runtoken
/// >>> enc = runtoken.encoding_for_model("gpt-4o")
/// >>> enc.count("Hello, world!")
/// 4
#[pyfunction]
fn encoding_for_model(model: &str) -> PyResult<PyTokenizer> {
    let enc_name = TokenizerRegistry::encoding_for_model(model);
    PyTokenizer::new(enc_name, None)
}

/// Quick count tokens for a given text and model.
///
/// >>> import runtoken
/// >>> runtoken.count("Hello, world!", model="gpt-4o")
/// 4
#[pyfunction]
#[pyo3(signature = (text, model="gpt-4o", encoding=None))]
fn count(text: &str, model: &str, encoding: Option<&str>) -> PyResult<usize> {
    let enc_name = encoding.unwrap_or_else(|| TokenizerRegistry::encoding_for_model(model));
    let tok = Tokenizer::new(enc_name)
        .map_err(|e| PyValueError::new_err(format!("Failed to load tokenizer: {}", e)))?;
    Ok(tok.count(text))
}

/// The runtoken Python module.
#[pymodule]
#[pyo3(name = "_runtoken")]
fn runtoken_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(get_encoding, m)?)?;
    m.add_function(wrap_pyfunction!(encoding_for_model, m)?)?;
    m.add_function(wrap_pyfunction!(count, m)?)?;
    Ok(())
}
