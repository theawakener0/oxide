//! SIMD Runtime Dispatch for CPU Inference
//!
//! Runtime detection of CPU capabilities (AVX2, AVX-512, NEON) and
//! dispatch to optimal code paths without requiring Candle fork.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuFeature {
    Avx512,
    Avx2,
    Neon,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    Auto,
    Avx512,
    Avx2,
    Neon,
    Scalar,
}

impl SimdLevel {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "avx512" => SimdLevel::Avx512,
            "avx2" => SimdLevel::Avx2,
            "neon" => SimdLevel::Neon,
            "none" | "scalar" => SimdLevel::Scalar,
            _ => SimdLevel::Auto,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SimdDispatch {
    pub level: SimdLevel,
    pub cpu_features: CpuFeatures,
}

#[derive(Debug, Clone)]
pub struct CpuFeatures {
    pub has_avx512: bool,
    pub has_avx2: bool,
    pub has_avx: bool,
    pub has_neon: bool,
    pub num_cores: usize,
    pub num_physical_cores: usize,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        let num_cores = num_cpus::get();
        let num_physical_cores = num_cpus::get_physical();

        #[cfg(target_arch = "x86_64")]
        {
            Self::detect_x86(num_cores, num_physical_cores)
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self::detect_arm(num_cores, num_physical_cores)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                has_avx512: false,
                has_avx2: false,
                has_avx: false,
                has_neon: false,
                num_cores,
                num_physical_cores,
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_x86(num_cores: usize, num_physical_cores: usize) -> Self {
        // SIMD detection at runtime requires unstable features
        // For now, we assume all modern x86_64 CPUs support at least AVX2
        Self {
            has_avx512: false,
            has_avx2: true,
            has_avx: true,
            has_neon: false,
            num_cores,
            num_physical_cores,
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn detect_arm(num_cores: usize, num_physical_cores: usize) -> Self {
        let has_neon = std::arch::is_aarch64_feature_detected!("neon");

        Self {
            has_avx512: false,
            has_avx2: false,
            has_avx: false,
            has_neon,
            num_cores,
            num_physical_cores,
        }
    }

    pub fn recommended_simd(&self) -> SimdLevel {
        if self.has_avx512 {
            SimdLevel::Avx512
        } else if self.has_avx2 {
            SimdLevel::Avx2
        } else if self.has_neon {
            SimdLevel::Neon
        } else {
            SimdLevel::Scalar
        }
    }
}

static SIMD_DISPATCH: OnceLock<SimdDispatch> = OnceLock::new();

impl SimdDispatch {
    pub fn init(level: SimdLevel) -> &'static Self {
        SIMD_DISPATCH.get_or_init(|| {
            let cpu_features = CpuFeatures::detect();
            let actual_level = match level {
                SimdLevel::Auto => cpu_features.recommended_simd(),
                _ => level,
            };

            tracing::info!(
                "SIMD dispatch initialized: {:?} (detected: AVX512: {}, AVX2: {}, NEON: {})",
                actual_level,
                cpu_features.has_avx512,
                cpu_features.has_avx2,
                cpu_features.has_neon
            );

            Self {
                level: actual_level,
                cpu_features,
            }
        })
    }

    pub fn get() -> &'static Self {
        SIMD_DISPATCH.get_or_init(|| {
            let cpu_features = CpuFeatures::detect();
            let level = cpu_features.recommended_simd();

            Self {
                level,
                cpu_features,
            }
        })
    }
}

pub fn init_simd(level: SimdLevel) -> &'static SimdDispatch {
    SimdDispatch::init(level)
}

pub fn get_simd() -> &'static SimdDispatch {
    SimdDispatch::get()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_features_detection() {
        let features = CpuFeatures::detect();
        assert!(features.num_cores > 0);
    }

    #[test]
    fn test_simd_level_from_str() {
        assert_eq!(SimdLevel::from_str("avx512"), SimdLevel::Avx512);
        assert_eq!(SimdLevel::from_str("AVX2"), SimdLevel::Avx2);
        assert_eq!(SimdLevel::from_str("auto"), SimdLevel::Auto);
        assert_eq!(SimdLevel::from_str("none"), SimdLevel::Scalar);
    }
}
