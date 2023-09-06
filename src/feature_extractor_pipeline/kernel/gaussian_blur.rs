use crate::wgsl::FVec4;
use crate::wgsl::declaration::FunctionVarDecl;
use crate::wgsl::expression::Expression;
use crate::wgsl::statement::AddAssignment;

use super::CenterOffset;

pub struct GaussianBlur{
    pub sigma: f32,
    pub max_offset: u32,
}

impl GaussianBlur{
    pub fn name(&self) -> String{
        let sigma_text = format!("{:.2}", self.sigma).replace(".", "_");
        return format!("gauss_blur_{sigma_text}")
    }
    pub fn accumulator_var(&self) -> FunctionVarDecl<FVec4>{
        let name = self.name();
        FunctionVarDecl {
            name: format!("{name}_accum"),
            initializer: Some(Expression::from(FVec4::zeros())),
        }
    }
    pub fn kernel_at(&self, center_offset: &CenterOffset) -> Option<f32>{
        if !self.is_defined_at(center_offset){
            return None
        }
        let x_2 = (center_offset.x * center_offset.x) as f32;
        let y_2 = (center_offset.y * center_offset.y) as f32;
        let two_sigma_2 = 2f32 * self.sigma * self.sigma;
        let exponent = - (x_2 + y_2) / two_sigma_2;

        return Some(
            (1f32 / (std::f32::consts::PI * two_sigma_2)) * std::f32::consts::E.powf(exponent)
        )
    }
    pub fn is_defined_at(&self, offset: &CenterOffset) -> bool{
        offset.x.unsigned_abs() <= self.max_offset &&
            offset.y.unsigned_abs() <= self.max_offset &&
            offset.z.unsigned_abs() <= self.max_offset
    }
    pub fn accumulate(
        &self, sample_var: &FunctionVarDecl<FVec4>, center_offset: &CenterOffset
    ) -> Option<AddAssignment<FVec4>>{
        let kernel_value = Expression::from(self.kernel_at(center_offset)?);
        Some(AddAssignment{
            assignee: self.accumulator_var(),
            value: kernel_value * &Expression::from(sample_var),
        })
    }
}

#[test]
fn test_gaussian(){
    let gb = GaussianBlur{sigma: 0.84089642, max_offset: 3};
    let radius: i32 = gb.max_offset.try_into().unwrap();
    let mut total: f32 = 0.0;

    for x in -(gb.max_offset as i32)..=(gb.max_offset as i32){
        for y in -radius..=radius{
            let Some(kernel_entry) = gb.kernel_at(&CenterOffset{x, y, z: 1}) else {
                continue;
            };
            print!(" {kernel_entry:.16} |");
            total += kernel_entry;
        }
        println!("");
    }

    println!("Total: {total:.16}");
}