use std::fmt::Display;


pub mod gaussian_blur;

pub struct CenterOffset{
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

pub struct AccumulatorVar(String);
impl AccumulatorVar{
    pub fn new(name: String) -> Self{
        return Self(name)
    }
}
impl Display for AccumulatorVar{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub struct SampleVar{
    pub var_name: String,
    pub center_offset: CenterOffset,
}
impl SampleVar{
    pub fn center_offset(&self) -> &CenterOffset{ return &self.center_offset }
}
impl Display for SampleVar{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.var_name)
    }
}

// pub struct CombinedFilter{}
// impl CombinedFilter{
//     fn produce_shader(){
//         use crate::feature_extractor_pipeline::kernel::gaussian_blur::GaussianBlur;

//         let gb = GaussianBlur::new(0.84089642, AccumulatorVar::new("gauss_accum".into()));
//         let radius: i32 = 3;

//         let mut total: f32 = 0.0;

//         for x in -radius..=radius{
//             for y in -radius..=radius{
//                 let sample_var = SampleVar{var_name: format!("sample"), center_offset: CenterOffset{x, y, z: 1}};
//                 let kernel_entry = gb.kernel_at(&sample_var);
//                 print!(" {kernel_entry:.16} |");
//                 total += kernel_entry
//             }
//             println!("");
//         }

//         println!("Total: {total:.16}");
//     }
// }