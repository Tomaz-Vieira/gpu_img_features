use super::{AccumulatorVar, SampleVar};

pub struct GaussianBlur{
    sigma: f32,
    accumulator_var: AccumulatorVar,
}

impl GaussianBlur{
    pub fn new(sigma: f32, accumulator_var: AccumulatorVar) -> Self{
        return Self{sigma, accumulator_var}
    }
    pub fn kernel_at(&self, sample_var: &SampleVar) -> f32{
        let center_offset = sample_var.center_offset();
        let x_2 = (center_offset.x * center_offset.x) as f32;
        let y_2 = (center_offset.y * center_offset.y) as f32;
        let two_sigma_2 = 2f32 * self.sigma * self.sigma;
        let exponent = - (x_2 + y_2) / two_sigma_2;

        return (1f32 / (std::f32::consts::PI * two_sigma_2)) * std::f32::consts::E.powf(exponent)
    }
    pub fn make_accumulate_statement(&self, sample_var: &SampleVar) -> String{
        let kernel_value = self.kernel_at(sample_var);
        let accumulator_var = &self.accumulator_var;
        format!("
            {accumulator_var} += {kernel_value} * {sample_var};
        ")
    }
}

#[test]
fn produce_shader(){
    use super::CenterOffset;

    let gb = GaussianBlur::new(0.84089642, AccumulatorVar::new("gauss_accum".into()));
    let radius: i32 = 3;

    let mut total: f32 = 0.0;

    for x in -radius..=radius{
        for y in -radius..=radius{
            let sample_var = SampleVar{var_name: format!("sample"), center_offset: CenterOffset{x, y, z: 1}};
            let kernel_entry = gb.kernel_at(&sample_var);
            print!(" {kernel_entry:.16} |");
            total += kernel_entry
        }
        println!("");
    }

    println!("Total: {total:.16}");
}