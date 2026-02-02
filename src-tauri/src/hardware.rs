use log::debug;
use serde::Serialize;
use std::process::Command;
use sysinfo::System;

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

#[derive(Debug, Serialize, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total: u64, // in bytes
}

#[derive(Debug, Serialize, Clone)]
pub struct HardwareInfo {
    pub cpu: String,
    pub ram_total: u64,
    pub gpus: Vec<GpuInfo>,
}

#[tauri::command]
pub async fn get_hardware_info_command() -> Result<HardwareInfo, String> {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu_name = sys
        .cpus()
        .get(0)
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown CPU".to_string());
    let ram_total = sys.total_memory();

    let mut gpus = Vec::new();

    #[cfg(target_os = "windows")]
    {
        // Gather GPU info using WMIC
        let output = Command::new("wmic")
            .args(&["path", "win32_videocontroller", "get", "name,AdapterRAM"])
            .output();

        if let Ok(out) = output {
            let s = String::from_utf8_lossy(&out.stdout);
            let lines: Vec<&str> = s.lines().collect();
            if lines.len() > 1 {
                for line in lines.iter().skip(1) {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        // AdapterRAM is usually the first or last column depending on wmic format
                        // We need to find which one is numeric
                        let mut name_parts = Vec::new();
                        let mut vram = 0u64;
                        for part in parts {
                            if let Ok(v) = part.parse::<u64>() {
                                vram = v;
                            } else {
                                name_parts.push(part);
                            }
                        }
                        let name = name_parts.join(" ");
                        if !name.is_empty() {
                            gpus.push(GpuInfo {
                                name,
                                vram_total: vram,
                            });
                        }
                    }
                }
            }
        }

        // If no GPUs found via WMIC or if we want more accurate VRAM for NVIDIA, use nvidia-smi
        if let Ok(out) = Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            let s = String::from_utf8_lossy(&out.stdout);
            for line in s.lines() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 2 {
                    let name = parts[0].trim().to_string();
                    if let Ok(v_mb) = parts[1].trim().parse::<u64>() {
                        let v_bytes = v_mb * 1024 * 1024;
                        // Replace or update existing GPU entry if name matches
                        if let Some(existing) = gpus
                            .iter_mut()
                            .find(|g| g.name.contains(&name) || name.contains(&g.name))
                        {
                            existing.vram_total = v_bytes;
                        } else {
                            gpus.push(GpuInfo {
                                name,
                                vram_total: v_bytes,
                            });
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        // On macOS, we can use system_profiler
        if let Ok(out) = Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
        {
            let s = String::from_utf8_lossy(&out.stdout);
            // Parsing system_profiler is complex, but let's do basic name extraction
            let mut current_name = String::new();
            for line in s.lines() {
                let line = line.trim();
                if line.ends_with(':')
                    && !line.contains("Displays")
                    && !line.contains("Graphics/Displays")
                {
                    current_name = line.trim_end_matches(':').to_string();
                }
                if line.contains("VRAM") && !current_name.is_empty() {
                    // Extract value
                    let parts: Vec<&str> = line.split(':').collect();
                    if parts.len() >= 2 {
                        let vram_str = parts[1].trim();
                        // Usually looks like "4 GB" or "8192 MB"
                        let vram_bytes = if vram_str.contains("GB") {
                            vram_str
                                .replace("GB", "")
                                .trim()
                                .parse::<u64>()
                                .unwrap_or(0)
                                * 1024
                                * 1024
                                * 1024
                        } else if vram_str.contains("MB") {
                            vram_str
                                .replace("MB", "")
                                .trim()
                                .parse::<u64>()
                                .unwrap_or(0)
                                * 1024
                                * 1024
                        } else {
                            0
                        };
                        gpus.push(GpuInfo {
                            name: current_name.clone(),
                            vram_total: vram_bytes,
                        });
                        current_name.clear();
                    }
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        // On Linux, try nvidia-smi first
        if let Ok(out) = Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            let s = String::from_utf8_lossy(&out.stdout);
            for line in s.lines() {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 2 {
                    let name = parts[0].trim().to_string();
                    if let Ok(v_mb) = parts[1].trim().parse::<u64>() {
                        let v_bytes = v_mb * 1024 * 1024;
                        gpus.push(GpuInfo {
                            name,
                            vram_total: v_bytes,
                        });
                    }
                }
            }
        }

        // If no NVIDIA, maybe add lspci check for AMD/Intel labeling?
        // But for now, Velox focuses on NVIDIA/Metal for training/inference
        if gpus.is_empty() {
            // Basic lspci check to at least show the name if possible
            if let Ok(out) = Command::new("lspci").output() {
                let s = String::from_utf8_lossy(&out.stdout);
                for line in s.lines() {
                    let lower = line.to_lowercase();
                    if (lower.contains("vga") || lower.contains("3d"))
                        && (lower.contains("nvidia")
                            || lower.contains("amd")
                            || lower.contains("intel"))
                    {
                        // Simple extraction: take everything after the first colon
                        // or just the whole line if parsing is hard
                        let name = if let Some(idx) = line.find(':') {
                            line[idx + 1..].trim().to_string()
                        } else {
                            line.to_string()
                        };

                        // We can't easily get VRAM from lspci, so set to 0 or leave it purely informational
                        gpus.push(GpuInfo {
                            name,
                            vram_total: 0,
                        });
                    }
                }
            }
        }
    }

    // Sort GPUs by VRAM (highest first)
    gpus.sort_by(|a, b| b.vram_total.cmp(&a.vram_total));

    Ok(HardwareInfo {
        cpu: cpu_name,
        ram_total,
        gpus,
    })
}
