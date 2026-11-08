//! Device / backend abstraction.
//!
//! llama.rs currently ships a complete CPU backend. GPU backends (Vulkan,
//! CUDA, Metal) are planned; the [`Backend`] enum and [`DeviceInfo`] struct
//! describe the execution target in a way that generalizes without pretending
//! that unimplemented backends exist.

use std::fmt;

/// An execution backend.
///
/// A backend is only ever reported as [`Backend::Vulkan`] etc. when the
/// corresponding feature is compiled in AND the device was successfully
/// initialized. There is no silent fallback that claims GPU execution while
/// running on the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// Portable scalar and SIMD CPU execution (always available).
    Cpu,
    /// Vulkan GPU execution (planned, feature-gated; not implemented yet).
    Vulkan,
    /// CUDA GPU execution (planned, feature-gated; not implemented yet).
    Cuda,
    /// Apple Metal GPU execution (planned, feature-gated; not implemented yet).
    Metal,
}

impl Backend {
    /// Human-readable backend name.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Vulkan => "Vulkan",
            Self::Cuda => "CUDA",
            Self::Metal => "Metal",
        }
    }

    /// Whether the backend is implemented in this build.
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Self::Cpu => true,
            Self::Vulkan | Self::Cuda | Self::Metal => false,
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Detected CPU capabilities used to select kernel implementations.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuFeatures {
    /// x86-64: AVX2 supported.
    pub avx2: bool,
    /// x86-64: AVX512F supported.
    pub avx512f: bool,
    /// x86-64: FMA supported.
    pub fma: bool,
    /// x86-64: F16C supported.
    pub f16c: bool,
    /// aarch64: NEON supported.
    pub neon: bool,
}

impl CpuFeatures {
    /// Probes the current CPU.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                fma: std::arch::is_x86_feature_detected!("fma"),
                f16c: std::arch::is_x86_feature_detected!("f16c"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is baseline on aarch64.
            Self {
                avx2: false,
                avx512f: false,
                fma: false,
                f16c: false,
                neon: true,
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::default()
        }
    }

    /// Short description of the best available vector ISA.
    #[must_use]
    pub fn describe(&self) -> String {
        #[cfg(target_arch = "x86_64")]
        {
            if self.avx512f {
                "AVX-512".to_string()
            } else if self.avx2 {
                "AVX2".to_string()
            } else if self.fma {
                "FMA".to_string()
            } else {
                "SSE2".to_string()
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if self.neon {
                "NEON".to_string()
            } else {
                "scalar".to_string()
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            "scalar".to_string()
        }
    }
}

/// Describes the device a model is loaded on.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Backend in use.
    pub backend: Backend,
    /// Number of compute threads.
    pub threads: usize,
    /// Detected CPU features (when the backend is CPU).
    pub cpu_features: CpuFeatures,
    /// Backend-specific description (e.g. CPU brand string).
    pub description: String,
}

impl DeviceInfo {
    /// Builds a CPU device description with detected features.
    #[must_use]
    pub fn cpu(threads: usize) -> Self {
        let cpu_features = CpuFeatures::detect();
        Self {
            backend: Backend::Cpu,
            threads,
            cpu_features,
            description: format!("CPU ({}, {} threads)", cpu_features.describe(), threads),
        }
    }
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_cpu_is_available_today() {
        assert!(Backend::Cpu.is_available());
        assert!(!Backend::Vulkan.is_available());
        assert!(!Backend::Cuda.is_available());
        assert!(!Backend::Metal.is_available());
    }

    #[test]
    fn cpu_info_detects() {
        let info = DeviceInfo::cpu(4);
        assert_eq!(info.backend, Backend::Cpu);
        assert!(info.cpu_features.avx2 || !info.cpu_features.avx2); // detection runs
        assert!(info.description.contains("CPU"));
    }
}
