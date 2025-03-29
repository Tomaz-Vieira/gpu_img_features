use encase::nalgebra::Vector2;

#[derive(Clone)]
pub struct GaussianBlur<const KSIDE: usize>{
    pub sigma: f32,
}

impl<const KSIDE: usize> GaussianBlur<KSIDE> {
    pub fn new(sigma: f32) -> Self{
        Self{
            sigma,
        }
    }
    pub const fn kernel_side_len(&self) -> usize{
        KSIDE
    }
    pub fn radius(&self) -> usize{
        (KSIDE - 1) / 2
    }
    pub fn required_size_in_bytes(&self) -> usize{
        self.kernel_side_len().pow(2) * std::mem::size_of::<f32>()
    }
    pub fn kernel_at(&self, center_offset: Vector2<i64>) -> f32 {
        use std::f32::consts::{PI, E};
        
        let x_2 = (center_offset.x * center_offset.x) as f32;
        let y_2 = (center_offset.y * center_offset.y) as f32;
        let two_sigma_2 = 2f32 * self.sigma * self.sigma;
        let exponent = -(x_2 + y_2) / two_sigma_2;

        return (1f32 / (PI * two_sigma_2)) * E.powf(exponent);
    }
    pub fn linear_idx_from_yx_offset(&self, offset: Vector2<i64>) -> usize{
        let y = offset.y + KSIDE as i64;
        let x = offset.x + KSIDE as i64;
        let out = y * self.kernel_side_len() as i64 + x;
        out.try_into().unwrap()
    }
    pub fn wgsl_linear_idx_from_yx_offset(&self, offset_var: &str) -> String{
        let radius = self.radius();
        let kernel_side_len = self.kernel_side_len();

        let y = format!("({offset_var}.y + {radius})");
        let x = format!("({offset_var}.x + {radius})");
        format!("( {y} * {kernel_side_len}  + {x} )")
    }
    pub fn fill_slice_yx(&self, buffer: &mut [[f32; KSIDE]; KSIDE]){
        let iradius: i64 = self.radius().try_into().unwrap();

        let mut total: f32 = 0.0;
        for y in -iradius..=iradius{
            for x in -iradius..=iradius{
                let val = self.kernel_at(Vector2::new(x, y));
                let fixed_y = usize::try_from(y + iradius).unwrap();
                let fixed_x = usize::try_from(x + iradius).unwrap();
                buffer[fixed_y][fixed_x] = val;
                total += val;
            }
        }
        assert!(1.0 - total < 0.001);
    }
}
