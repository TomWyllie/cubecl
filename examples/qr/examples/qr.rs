fn main() {
    #[cfg(feature = "cuda")]
    qr::launch::<cubecl::cuda::CudaRuntime>(&Default::default());
    #[cfg(feature = "wgpu")]
    qr::launch::<cubecl::wgpu::WgpuRuntime>(&Default::default());
}
