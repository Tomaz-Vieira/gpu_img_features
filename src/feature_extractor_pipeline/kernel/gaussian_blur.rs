use super::{AccumulatorVar, SampleVar, CenterOffset};

pub struct GaussianBlur{
    pub sigma: f32,
    pub max_offset: u8,
}

impl GaussianBlur{
    pub fn make_accumulator_var(&self) -> AccumulatorVar{
        return AccumulatorVar(format!("GaussianBlur_accum_{}", self.sigma.to_string().replace(".", "_")))
    }
    pub fn kernel_at(&self, center_offset: &CenterOffset) -> f32{
        let x_2 = (center_offset.x * center_offset.x) as f32;
        let y_2 = (center_offset.y * center_offset.y) as f32;
        let two_sigma_2 = 2f32 * self.sigma * self.sigma;
        let exponent = - (x_2 + y_2) / two_sigma_2;

        return (1f32 / (std::f32::consts::PI * two_sigma_2)) * std::f32::consts::E.powf(exponent)
    }
    pub fn is_defined_at(&self, offset: &CenterOffset) -> bool{
        offset.x.unsigned_abs() <= self.max_offset &&
            offset.y.unsigned_abs() <= self.max_offset &&
            offset.z.unsigned_abs() <= self.max_offset
    }
    pub fn make_accumulate_statement(
        &self, accumulator_var: &AccumulatorVar, sample_var: &SampleVar, center_offset: &CenterOffset
    ) -> Option<String>{
        if !self.is_defined_at(center_offset){
            return None
        }
        let kernel_value = self.kernel_at(center_offset);
        Some(format!("{accumulator_var} += {kernel_value:.20e} * {sample_var};"))
    }
}

#[test]
fn test_gaussian(){
    use super::CenterOffset;

    let gb = GaussianBlur{sigma: 0.84089642, max_offset: 3};
    let radius: i8 = gb.max_offset.try_into().unwrap();
    let mut total: f32 = 0.0;

    for x in -(gb.max_offset as i8)..=(gb.max_offset as i8){
        for y in -radius..=radius{
            let kernel_entry = gb.kernel_at(&CenterOffset{x, y, z: 1});
            print!(" {kernel_entry:.16} |");
            total += kernel_entry;
        }
        println!("");
    }

    println!("Total: {total:.16}");
}