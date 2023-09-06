pub mod combined_filters;


pub mod gaussian_blur;

pub struct CenterOffset{
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

// pub trait IKernelGenerator{
//     fn kernel_at(&self, sample_var: &SampleVar) -> f32;
//     fn make_accumulate_statement(&self, sample_var: &SampleVar) -> String;
// }
