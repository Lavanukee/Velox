use log::debug;
use serde::Serialize;
use std::process::Command;

#[derive(Debug, Serialize, Clone)]
pub enum ComputeBackend {
    Cuda,
    #[allow(dead_code)]
    Metal,
    Cpu,
}

pub fn detect_backend() -> ComputeBackend {
    #[cfg(target_os = "macos")]
    {
        // Check for Apple Silicon
        let output = Command::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output()
            .ok();

        if let Some(out) = output {
            let s = String::from_utf8_lossy(&out.stdout).to_lowercase();
            if s.contains("apple") {
                return ComputeBackend::Metal;
            }
        }
        return ComputeBackend::Cpu; // Intel Mac
    }

    #[cfg(target_os = "windows")]
    {
        // 1. PRIORITY CHECK: Try nvidia-smi (The gold standard for checking CUDA availability)
        debug!("Detecting Backend: Attempting nvidia-smi check...");
        match Command::new("nvidia-smi").arg("-L").output() {
            Ok(output) => {
                let s = String::from_utf8_lossy(&output.stdout).to_lowercase();
                debug!("nvidia-smi output: {}", s);
                if s.contains("gpu") || s.contains("nvidia") {
                    debug!("Backend Decision: CUDA (via nvidia-smi)");
                    return ComputeBackend::Cuda;
                }
            }
            Err(e) => debug!("nvidia-smi check failed/not found: {}", e),
        }

        // 2. SECONDARY CHECK: Check Environment Variables
        if std::env::var("CUDA_PATH").is_ok() {
            debug!("Backend Decision: CUDA (via CUDA_PATH env var)");
            return ComputeBackend::Cuda;
        }

        // 3. FALLBACK CHECK: WMIC
        debug!("Detecting Backend: Attempting WMIC check...");
        let output = Command::new("wmic")
            .args(&["path", "win32_videocontroller", "get", "name"])
            .output()
            .ok();

        if let Some(out) = output {
            let s = String::from_utf8_lossy(&out.stdout).to_lowercase();
            debug!("WMIC output: {}", s);
            if s.contains("nvidia") {
                debug!("Backend Decision: CUDA (via WMIC)");
                return ComputeBackend::Cuda;
            }
        }

        // 4. Default to CPU
        debug!("Backend Decision: CPU (No NVIDIA indicators found)");
        return ComputeBackend::Cpu;
    }

    #[cfg(target_os = "linux")]
    {
        // Check 1: nvidia-smi
        if let Ok(output) = Command::new("nvidia-smi").arg("-L").output() {
            let s = String::from_utf8_lossy(&output.stdout).to_lowercase();
            if s.contains("nvidia") {
                return ComputeBackend::Cuda;
            }
        }

        // Check 2: lspci
        let output = Command::new("lspci").output().ok();
        if let Some(out) = output {
            let s = String::from_utf8_lossy(&out.stdout).to_lowercase();
            if s.contains("nvidia") {
                return ComputeBackend::Cuda;
            }
        }
        return ComputeBackend::Cpu;
    }
}
