pub mod combined_filters;

use std::fmt::Display;




pub mod gaussian_blur;

pub struct CenterOffset{
    pub x: i8,
    pub y: i8,
    pub z: i8,
}

pub struct AccumulatorVar(pub String);
impl Display for AccumulatorVar{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct SampleVar(pub String);
impl Display for SampleVar{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub trait IKernelGenerator{
    fn kernel_at(&self, sample_var: &SampleVar) -> f32;
    fn make_accumulate_statement(&self, sample_var: &SampleVar) -> String;
}
