# Project ASTRA Solana Integration (Web3 & DePIN)

## Architectural Flow: Off-Chain AI Guardrail Node

Project ASTRA is engineered as an **Off-Chain AI Guardrail Node**. It operates seamlessly within a Solana Agent Kit pipeline or an Anchor-based DePIN architecture.

**The Workflow:**
1. A Solana AI Agent RPC call receives a request to process data or execute an AI model step.
2. Before sending raw activation tensors to the next layer (or committing any state on-chain), the intermediate states are passed through the ASTRA guardrail (`astra_core.py`).
3. ASTRA applies Float64 Nullspace Projection to filter adversarial noise and bounds OOD activations.
4. A deterministic execution digest (SHA-256 state hash) is computed based on the transformed tensors.
5. The cleaned tensor state is used to generate the final prediction or output.
6. The off-chain worker signs the digest and submits it to a Solana Anchor Program, registering a verified DePIN AI computation proof on-chain.

## Rust / Solana Agent Kit Wrapper Example

Below is a conceptual Rust wrapper demonstrating how a Solana program (or off-chain Rust service) interfaces with the Python `astra_core.py` engine via FFI or a localized IPC server.

```rust
use reqwest::blocking::Client;
use serde_json::json;
use sha2::{Sha256, Digest};

/// Struct representing the Off-Chain ASTRA Validator Node
pub struct AstraValidatorNode {
    pub endpoint_url: String,
}

impl AstraValidatorNode {
    pub fn new(url: &str) -> Self {
        Self { endpoint_url: url.to_string() }
    }

    /// Submits a raw tensor state to ASTRA and retrieves the secured tensor and state hash
    pub fn execute_guardrail(&self, raw_tensor_data: Vec<f64>) -> Result<(Vec<f64>, String), String> {
        let client = Client::new();

        // Serialize tensor state
        let payload = json!({
            "tensor": raw_tensor_data,
            "dimensions": [1, 32, 64, 64] // Example Batch x Channels x H x W
        });

        // Call Off-Chain ASTRA Node
        let response = client.post(&self.endpoint_url)
            .json(&payload)
            .send()
            .map_err(|e| e.to_string())?;

        let result_json: serde_json::Value = response.json().map_err(|e| e.to_string())?;

        let safe_tensor: Vec<f64> = serde_json::from_value(result_json["safe_tensor"].clone())
            .map_err(|e| e.to_string())?;

        let proof_hash = result_json["proof_hash"].as_str().unwrap_or("").to_string();

        Ok((safe_tensor, proof_hash))
    }
}
```

## Cryptographic Proof Log (Verifiable Computation)

To facilitate verifiable DePIN computation, ASTRA computes a deterministic digest of its transformation outputs.

In Python (`astra_core.py` usage):
```python
import hashlib
import torch

def generate_astra_proof_log(projected_tensor: torch.Tensor) -> str:
    """
    Computes a deterministic SHA-256 state hashing of the nullspace
    transformation to log it for on-chain verifiable computation.
    """
    # Ensure tensor is strictly Float64 on CPU for deterministic hashing
    tensor_bytes = projected_tensor.detach().cpu().to(torch.float64).numpy().tobytes()

    sha256_hash = hashlib.sha256()
    sha256_hash.update(tensor_bytes)
    return sha256_hash.hexdigest()
```
The resulting `proof_hash` is appended to the Solana Transaction instruction as metadata, proving the execution passed through the mathematical constraints of the ASTRA deflector prior to on-chain state settlement.