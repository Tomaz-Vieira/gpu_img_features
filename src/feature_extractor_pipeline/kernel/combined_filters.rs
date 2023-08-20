use wgpu::Extent3d;

use crate::feature_extractor_pipeline::kernel::{SampleVar, CenterOffset, AccumulatorVar};

use super::gaussian_blur::GaussianBlur;

pub struct CombinedFilters<'kernels>{
    pub kernels: &'kernels [GaussianBlur],
    pub input_extent: Extent3d,
}
impl CombinedFilters<'_>{
    pub fn produce_shader(&self) -> String{
        let mut shader_code = String::with_capacity(1024 * 1024);

        let accumulator_vars: Vec<AccumulatorVar> = self.kernels.iter().map(|kernel| kernel.make_accumulator_var()).collect();
        let sample_var = SampleVar(format!("sample"));
        let radius: i8 = self.kernels.iter().map(|k| k.max_offset).max().unwrap().try_into().unwrap();

        accumulator_vars.iter().for_each(|acc_var|{
            shader_code += &format!("let {acc_var}: f32 = 0.0;\n");
        });
        shader_code += "\n";

        for y in -radius..=radius{
            for x in -radius..=radius{
                let center_offset = CenterOffset{x, y, z: 0};
                //FIXME: use a sampler instead
                let clamped_x: i32 = (i32::from(x)).clamp(0, (self.input_extent.width - 1).try_into().unwrap());
                let clamped_y: i32 = (i32::from(y)).clamp(0, (self.input_extent.height - 1).try_into().unwrap());
                shader_code += &format!("{sample_var} = textureLoad(some_input, vec2i({clamped_x}, {clamped_y}), 0);\n");
                for (kernel_index, kernel) in self.kernels.iter().enumerate(){
                    match kernel.make_accumulate_statement(&accumulator_vars[kernel_index], &sample_var, &center_offset){
                        None => continue,
                        Some(shader_line) => {
                            shader_code += &shader_line;
                            shader_code += "\n";
                        }
                    }
                }
            }
        }
        return shader_code
    }
}

#[test]
fn produce_shader(){
    let shader_code = CombinedFilters{
        kernels: &[
            GaussianBlur{sigma: 0.3, max_offset: 3},
            GaussianBlur{sigma: 0.7, max_offset: 3},
        ],
        input_extent: Extent3d { width: 3, height: 3, depth_or_array_layers: 1 }
    }.produce_shader();
    println!("Shader code:\n{shader_code}");
}