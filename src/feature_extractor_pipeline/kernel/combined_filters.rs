use crate::feature_extractor_pipeline::kernel::{SampleVar, CenterOffset, AccumulatorVar};

use super::gaussian_blur::GaussianBlur;

pub struct CombinedFilters{}
impl CombinedFilters{
    fn produce_shader(kernels: &[GaussianBlur]){
        let mut shader_code = String::with_capacity(1024 * 1024);

        let accumulator_vars: Vec<AccumulatorVar> = kernels.iter().map(|kernel| kernel.make_accumulator_var()).collect();
        let sample_var = SampleVar(format!("sample"));
        let radius: i8 = kernels.iter().map(|k| k.max_offset).max().unwrap().try_into().unwrap();

        accumulator_vars.iter().for_each(|acc_var|{
            shader_code += &format!("let {acc_var}: f32 = 0.0;\n");
        });
        shader_code += "\n";

        for y in -radius..=radius{
            for x in -radius..=radius{
                let center_offset = CenterOffset{x, y, z: 0};
                shader_code += &format!("{sample_var} = textureLoad(some_input, vec2i({x}, {y}), 0);\n");
                for (kernel_index, kernel) in kernels.iter().enumerate(){
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

        println!("Shader code:\n{shader_code}");
    }
}

#[test]
fn produce_shader(){
    CombinedFilters::produce_shader(&[
        GaussianBlur{sigma: 0.3, max_offset: 3},
        GaussianBlur{sigma: 0.7, max_offset: 3},
    ])
}